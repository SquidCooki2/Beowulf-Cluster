import argparse
import json
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim

from network.transceiver import TensorTransceiver
from models.registry import get_spec
from engine.scheduler import Pipeline1F1BScheduler
from engine.runner import run_train, run_infer

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [Stage %(process)d] [%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Run 1F1B Pipeline Stage Daemon")
    parser.add_argument("--stage", type=int, required=True, help="Stage ID (0 to 4)")
    parser.add_argument("--config", type=str, default="/shared/pipe1f1b/config.json", help="Path to config.json")
    parser.add_argument("--model", type=str, default=None, help="Registered model name (default: config 'model')")
    parser.add_argument("--weights-dir", type=str, default=None,
                        help="Dir with stage{i}.pt saved by training.train_local (optional)")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Skip ImageNet weights (avoids a download; fine for pure timing runs)")
    parser.add_argument("--mode", choices=["synthetic", "train", "infer"], default="synthetic",
                        help="synthetic: one random-data iteration (connectivity self-test)")
    parser.add_argument("--data-dir", type=str, default="/shared/data", help="Dir holding malaria_128.npz")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="Use N samples total (must match on all stages)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--run-id", type=str, default="engine-run", help="Must be identical on all stages")
    parser.add_argument("--platform-label", type=str, default="engine")
    parser.add_argument("--latency-samples", type=int, default=200)
    parser.add_argument("--infer-batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    args = parser.parse_args()

    # Load cluster config
    try:
        with open(args.config, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        with open("config.json", "r") as f:
            config = json.load(f)

    stage_id = args.stage
    stage_info = config["stages"][stage_id]
    model_name = args.model or config.get("model", "malaria_mobilenet")
    spec = get_spec(model_name)
    logging.info(f"Initializing Stage {stage_id} on {stage_info['hostname']} ({stage_info['ip']}) - Device: {stage_info['device_type']} - Model: {model_name}")

    # Sliced models
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    args.model_name = model_name
    all_stages = spec.create_stages(len(config["stages"]), pretrained=not args.no_pretrained)
    local_stage_model = all_stages[stage_id]
    if args.weights_dir:
        local_stage_model.load_state_dict(torch.load(os.path.join(args.weights_dir, f"stage{stage_id}.pt")))
        logging.info(f"Loaded weights from {args.weights_dir}")
    
    # Optimizer and Loss
    optimizer = optim.Adam([p for p in local_stage_model.parameters() if p.requires_grad], lr=config["learning_rate"])
    last = len(config["stages"]) - 1
    criterion = (nn.CrossEntropyLoss() if spec.loss == "ce" else nn.MSELoss()) if stage_id == last else None

    # Setup ZeroMQ communication
    transceiver = TensorTransceiver(stage_id=stage_id, config=config)

    # Instantiate 1F1B Scheduler
    scheduler = Pipeline1F1BScheduler(
        stage_id=stage_id,
        num_stages=len(config["stages"]),
        num_microbatches=config["num_microbatches"],
        model_stage=local_stage_model,
        transceiver=transceiver,
        optimizer=optimizer,
        criterion=criterion
    )

    logging.info(f"Stage {stage_id} initialized successfully! Ready for 1F1B execution.")

    if args.mode == "train":
        run_train(stage_id, len(config["stages"]), local_stage_model, scheduler, transceiver, config, args)
        return
    if args.mode == "infer":
        run_infer(stage_id, len(config["stages"]), local_stage_model, transceiver, config, args)
        return

    # Synthetic self-test iteration
    if stage_id == 0:
        logging.info("Starting synthetic test batch across cluster...")
        synthetic_images = [torch.randn(config["micro_batch_size"], *spec.input_shape) for _ in range(config["num_microbatches"])]
        scheduler.run_1f1b_epoch(dataloader_iterator=iter(synthetic_images))
        logging.info("Synthetic training iteration completed successfully across pipeline.")
    elif stage_id == last:
        if spec.loss == "ce":
            synthetic_targets = [torch.randint(0, spec.num_outputs, (config["micro_batch_size"],)) for _ in range(config["num_microbatches"])]
        else:
            synthetic_targets = [torch.randn(config["micro_batch_size"], spec.num_outputs) for _ in range(config["num_microbatches"])]
        scheduler.run_1f1b_epoch(target_iterator=iter(synthetic_targets))
        logging.info(f"Stage {last}: Processed loss and gradients for synthetic batch.")
    else:
        scheduler.run_1f1b_epoch()

if __name__ == "__main__":
    main()
