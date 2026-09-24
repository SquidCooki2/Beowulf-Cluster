import torch
import logging

class Pipeline1F1BScheduler:
    """
    1F1B (One-Forward-One-Backward) Pipeline Parallelism Scheduler.
    Maintains a strict FIFO buffer of activations, bounding memory to O(p) stages.
    """
    def __init__(self, stage_id: int, num_stages: int, num_microbatches: int,
                 model_stage: torch.nn.Module, transceiver, optimizer=None, criterion=None):
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches
        self.model = model_stage
        self.transceiver = transceiver
        self.optimizer = optimizer
        self.criterion = criterion

        # FIFO buffer bounding resident RAM activations to at most num_stages
        self.saved_activations = []

    def forward_step(self, input_tensor=None):
        """Executes forward pass for one micro-batch and forwards output downstream."""
        if self.stage_id > 0:
            input_tensor = self.transceiver.recv_forward()

        input_tensor.requires_grad_()
        output_tensor = self.model(input_tensor)

        # Save for backward pass
        self.saved_activations.append((input_tensor, output_tensor))

        if self.stage_id < self.num_stages - 1:
            self.transceiver.send_forward(output_tensor)
        return output_tensor

    def backward_step(self, target_labels=None):
        """Pops oldest activation from FIFO, executes local backward pass, and sends gradient upstream."""
        input_tensor, output_tensor = self.saved_activations.pop(0)

        if self.stage_id == self.num_stages - 1:
            # Terminal stage: compute loss
            loss = self.criterion(output_tensor, target_labels)
            loss.backward()
            incoming_grad = input_tensor.grad
        else:
            grad_output = self.transceiver.recv_backward()
            torch.autograd.backward(output_tensor, grad_tensors=grad_output)
            incoming_grad = input_tensor.grad

        if self.stage_id > 0:
            self.transceiver.send_backward(incoming_grad)

    def run_1f1b_epoch(self, dataloader_iterator=None, target_iterator=None):
        """
        Executes the 3-phase 1F1B pipeline cycle:
        1. Warmup: Injects micro-batches to fill the pipeline.
        2. Steady State: 1-Forward followed by 1-Backward.
        3. Cooldown: Drains remaining backward passes.
        """
        num_warmup = min(self.num_stages - self.stage_id, self.num_microbatches)

        # 1. Warmup Phase
        for _ in range(num_warmup):
            if self.stage_id == 0:
                batch_data = next(dataloader_iterator)
                self.forward_step(batch_data)
            elif self.stage_id == self.num_stages - 1:
                pass
            else:
                self.forward_step()

        # 2. 1F1B Steady State Phase
        remaining_fwd = self.num_microbatches - num_warmup
        for _ in range(remaining_fwd):
            if self.stage_id == 0:
                batch_data = next(dataloader_iterator)
                self.forward_step(batch_data)
            else:
                self.forward_step()

            targets = next(target_iterator) if self.stage_id == self.num_stages - 1 else None
            self.backward_step(targets)

        # 3. Cooldown Phase
        for _ in range(num_warmup):
            targets = next(target_iterator) if self.stage_id == self.num_stages - 1 else None
            self.backward_step(targets)

        # Step optimizer and reset gradients
        if self.optimizer:
            self.optimizer.step()
            self.optimizer.zero_grad()
