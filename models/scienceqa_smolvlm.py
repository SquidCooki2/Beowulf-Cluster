"""SmolVLM-256M-Instruct (`HuggingFaceTB/SmolVLM-256M-Instruct`) sliced into N pipeline stages.

Designed for the 5-node AgriBeowulf-5Node cluster (8 GB RAM / 128 GB SSD per node):
  - Stage 0 (`thinkcentre1`, Lenovo M900):
      Unpacks `(B, 4, 128, 128)` -> RGB image `(B, 3, 128, 128)` + prompt token IDs `(B, 60)`,
      runs the frozen SigLIP vision encoder + trainable pixel-shuffle connector -> `(B, 4, 576)`,
      concatenates with token embeddings `(B, 60, 576)` -> `(B, 64, 576)`, and runs the first
      3 SmolLM2 decoder layers.
  - Stages 1..3 (`prodesk1..3`, HP ProDesk 400 G5):
      Each runs 8 SmolLM2 decoder layers on the `(B, 64, 576)` sequence tensor.
  - Stage 4 (`thinkcentre2`, Lenovo M900):
      Runs the final 3 SmolLM2 decoder layers + RMSNorm, selects the final `"Answer:"` token
      representation `[:, -1, :]`, and projects to 5 answer-choice logits (`A..E`) initialized
      from SmolVLM's pretrained `lm_head`.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.scienceqa import CHOICE_LETTERS, MAX_TEXT_LEN, MODEL_ID, NUM_CHOICES

HIDDEN_SIZE = 576
INTERMEDIATE_SIZE = 1536
NUM_DEC_LAYERS = 30
NUM_HEADS = 9
VOCAB_SIZE = 49152
NUM_VISUAL_TOKENS = 4


def _layer_counts(total_layers: int, num_stages: int):
    """Return per-stage decoder layer counts, weighting middle ProDesk stages higher for 5 stages."""
    if num_stages == 1:
        return [total_layers]
    if num_stages == 5 and total_layers == 30:
        return [3, 8, 8, 8, 3]
    base, extra = divmod(total_layers, num_stages)
    return [base + (1 if i < extra else 0) for i in range(num_stages)]


def _causal_mask(batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """4D additive causal attention mask `(B, 1, S, S)` compatible with all transformers versions."""
    mask = torch.full((seq_len, seq_len), -1e4, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)


class _FallbackVisionConnector(nn.Module):
    """Lightweight SigLIP-shaped vision stem + pixel-shuffle projector used when `--no-pretrained`
    is run without `transformers` installed."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=16, stride=16),
            nn.GELU(),
            nn.Conv2d(192, 384, kernel_size=1),
        )
        for p in self.stem.parameters():
            p.requires_grad = False
        self.stem.eval()
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(2),
            nn.Linear(384, hidden_size, bias=False),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.stem(pixel_values)
        feat = self.proj[:2](feat).transpose(1, 2)  # (B, 4, 384)
        return self.proj[2](feat)                  # (B, 4, hidden_size)


class _FallbackDecoderLayer(nn.Module):
    """Compact Pre-LN Causal Transformer block matching SmolLM2-135M dimensions (576d, 9 heads)."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE, intermediate_size: int = INTERMEDIATE_SIZE,
                 num_heads: int = NUM_HEADS):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, bias=False)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(intermediate_size, hidden_size, bias=False),
        )

    def forward(self, x: torch.Tensor, causal_mask_2d: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask_2d, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderSliceRunner(nn.Module):
    """Runs a contiguous slice of `LlamaDecoderLayer` (or fallback) blocks on `(B, S, D)`."""

    def __init__(self, layers: nn.ModuleList, rotary_emb: nn.Module = None):
        super().__init__()
        self.layers = layers
        self.rotary_emb = copy.deepcopy(rotary_emb) if rotary_emb is not None else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if len(self.layers) == 0:
            return hidden_states
        b, seq_len, _ = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype

        if isinstance(self.layers[0], _FallbackDecoderLayer):
            mask_2d = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype), diagonal=1
            )
            for layer in self.layers:
                hidden_states = layer(hidden_states, mask_2d)
            return hidden_states

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(b, seq_len)
        causal_mask = _causal_mask(b, seq_len, device, dtype)
        pos_emb = self.rotary_emb(hidden_states, position_ids) if self.rotary_emb is not None else None

        for layer in self.layers:
            kwargs = {"attention_mask": causal_mask, "position_ids": position_ids}
            if pos_emb is not None:
                kwargs["position_embeddings"] = pos_emb
            out = layer(hidden_states, **kwargs)
            hidden_states = out[0] if isinstance(out, tuple) else out
        return hidden_states


class SmolVLMStage0(nn.Module):
    """Stage 0 (`thinkcentre1`): Unpacks image + tokens, runs frozen SigLIP + trainable connector
    + token embeddings, fuses into `(B, 64, 576)`, and runs the initial decoder layers."""

    def __init__(self, vision_model: nn.Module, connector: nn.Module, embed_tokens: nn.Embedding,
                 decoder_runner: DecoderSliceRunner, vocab_size: int = VOCAB_SIZE,
                 max_text_len: int = MAX_TEXT_LEN):
        super().__init__()
        self.vision_model = vision_model
        self.connector = connector
        self.embed_tokens = embed_tokens
        self.decoder_runner = decoder_runner
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len

        if self.vision_model is not None:
            for p in self.vision_model.parameters():
                p.requires_grad = False
            self.vision_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.vision_model is not None:
            self.vision_model.eval()
        return self

    def _encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.vision_model is None:
            return self.connector(pixel_values)
        with torch.no_grad():
            vis_out = self.vision_model(pixel_values=pixel_values)
            vis_feat = vis_out.last_hidden_state if hasattr(vis_out, "last_hidden_state") else vis_out[0]
        vis_tokens = self.connector(vis_feat)
        if vis_tokens.shape[1] != NUM_VISUAL_TOKENS:
            vis_tokens = F.adaptive_avg_pool1d(vis_tokens.transpose(1, 2), NUM_VISUAL_TOKENS).transpose(1, 2)
        return vis_tokens

    def forward(self, packed_input: torch.Tensor) -> torch.Tensor:
        pixel_values = packed_input[:, :3, :, :]
        raw_tokens = packed_input[:, 3, 0, :self.max_text_len]
        input_ids = raw_tokens.detach().long().clamp(0, self.vocab_size - 1)

        vis_embeds = self._encode_image(pixel_values)
        txt_embeds = self.embed_tokens(input_ids)
        hidden_states = torch.cat([vis_embeds, txt_embeds], dim=1)
        return self.decoder_runner(hidden_states)


class SmolVLMLastStage(nn.Module):
    """Terminal Stage (`thinkcentre2`): Runs final decoder layers + RMSNorm + 5-choice LM Head."""

    def __init__(self, decoder_runner: DecoderSliceRunner, norm: nn.Module, answer_head: nn.Linear):
        super().__init__()
        self.decoder_runner = decoder_runner
        self.norm = norm
        self.answer_head = answer_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.decoder_runner(hidden_states)
        hidden_states = self.norm(hidden_states)
        last_token = hidden_states[:, -1, :]
        return self.answer_head(last_token)


def _init_answer_head_from_lm_head(answer_head: nn.Linear, lm_head: nn.Linear):
    """Initialize the 5-choice ('A'..'E') projection rows from SmolVLM's pretrained `lm_head`."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        ids = []
        for letter in CHOICE_LETTERS[:answer_head.out_features]:
            enc = tok.encode(letter, add_special_tokens=False)
            ids.append(enc[-1] if enc else ord(letter))
        with torch.no_grad():
            answer_head.weight.copy_(lm_head.weight[ids])
        return
    except Exception:
        pass
    with torch.no_grad():
        rows = [ord(c) % lm_head.weight.shape[0] for c in CHOICE_LETTERS[:answer_head.out_features]]
        answer_head.weight.copy_(lm_head.weight[rows])


def _load_hf_components(pretrained: bool):
    """Load SmolVLM-256M components via Hugging Face `transformers`, or return None if unavailable."""
    try:
        import transformers
    except ImportError:
        if pretrained:
            raise ImportError(
                "Hugging Face `transformers` is required for pretrained SmolVLM-256M. "
                "Install with: pip install transformers>=4.46.0"
            )
        return None

    if pretrained:
        AutoVLM = getattr(transformers, "AutoModelForImageTextToText", None) or getattr(
            transformers, "AutoModelForVision2Seq"
        )
        base = AutoVLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    else:
        try:
            from transformers import AutoConfig, Idefics3ForConditionalGeneration
            cfg = AutoConfig.from_pretrained(MODEL_ID)
            base = Idefics3ForConditionalGeneration(cfg)
        except Exception:
            return None

    inner = getattr(base, "model", base)
    text_model = inner.text_model
    return {
        "vision_model": inner.vision_model,
        "connector": inner.connector,
        "embed_tokens": text_model.embed_tokens,
        "layers": list(text_model.layers),
        "norm": text_model.norm,
        "rotary_emb": getattr(text_model, "rotary_emb", None),
        "lm_head": getattr(base, "lm_head", None),
        "hidden_size": text_model.config.hidden_size,
        "vocab_size": text_model.config.vocab_size,
    }


def create_smolvlm_stages(num_stages: int = 5, num_classes: int = NUM_CHOICES, pretrained: bool = True):
    """Create `num_stages` sequential `nn.Module` stages for SmolVLM-256M on ScienceQA."""
    hf = _load_hf_components(pretrained=pretrained)

    if hf is not None:
        vision_model = hf["vision_model"]
        connector = hf["connector"]
        embed_tokens = hf["embed_tokens"]
        all_layers = hf["layers"]
        norm = hf["norm"]
        rotary_emb = hf["rotary_emb"]
        hidden_size = hf["hidden_size"]
        vocab_size = hf["vocab_size"]
        answer_head = nn.Linear(hidden_size, num_classes, bias=False)
        if hf["lm_head"] is not None:
            _init_answer_head_from_lm_head(answer_head, hf["lm_head"])
    else:
        vision_model = None
        connector = _FallbackVisionConnector(HIDDEN_SIZE)
        embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        all_layers = [_FallbackDecoderLayer(HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_HEADS)
                      for _ in range(NUM_DEC_LAYERS)]
        norm = nn.LayerNorm(HIDDEN_SIZE)
        rotary_emb = None
        vocab_size = VOCAB_SIZE
        answer_head = nn.Linear(HIDDEN_SIZE, num_classes, bias=False)

    counts = _layer_counts(len(all_layers), num_stages)
    slices, idx = [], 0
    for c in counts:
        slices.append(nn.ModuleList(all_layers[idx:idx + c]))
        idx += c

    if num_stages == 1:
        s0 = SmolVLMStage0(vision_model, connector, embed_tokens,
                           DecoderSliceRunner(slices[0], rotary_emb), vocab_size=vocab_size)
        s_last = SmolVLMLastStage(DecoderSliceRunner(nn.ModuleList(), rotary_emb), norm, answer_head)
        return [nn.Sequential(s0, s_last)]

    stages = [
        SmolVLMStage0(vision_model, connector, embed_tokens,
                      DecoderSliceRunner(slices[0], rotary_emb), vocab_size=vocab_size)
    ]
    for s in range(1, num_stages - 1):
        stages.append(DecoderSliceRunner(slices[s], rotary_emb))
    stages.append(SmolVLMLastStage(DecoderSliceRunner(slices[-1], rotary_emb), norm, answer_head))
    return stages
