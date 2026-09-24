import argparse
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim

from network.transceiver import TensorTransceiver
from models.leaf_resnet import create_resnet18_stages
from engine.scheduler import Pipeline1F1BScheduler

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [Stage %(process)d] [%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Run 1F1B Pipeline Stage Daemon")
    parser.add_argument("--stage", type=int, required=True, help="Stage ID (0 to 4)")
    parser.add_argument("--config", type=str, default="/shared/pipe1f1b/config.json", help="Path to config.json")
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
    logging.info(f"Initializing Stage {stage_id} on {stage_info['hostname']} ({stage_info['ip']}) - Device: {stage_info['device_type']}")

    # Sliced models
    all_stages = create_resnet18_stages(num_classes=config["num_classes"])
    local_stage_model = all_stages[stage_id]
    
    # Optimizer and Loss
    optimizer = optim.Adam(local_stage_model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss() if stage_id == 4 else None

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

    # Synthetic self-test iteration
    if stage_id == 0:
        logging.info("Starting synthetic test batch across cluster...")
        synthetic_images = [torch.randn(config["micro_batch_size"], 3, 224, 224) for _ in range(config["num_microbatches"])]
        scheduler.run_1f1b_epoch(dataloader_iterator=iter(synthetic_images))
        logging.info("Synthetic training iteration completed successfully across pipeline.")
    elif stage_id == 4:
        synthetic_targets = [torch.randint(0, config["num_classes"], (config["micro_batch_size"],)) for _ in range(config["num_microbatches"])]
        scheduler.run_1f1b_epoch(target_iterator=iter(synthetic_targets))
        logging.info("Stage 4: Processed loss and gradients for synthetic batch.")
    else:
        scheduler.run_1f1b_epoch()

if __name__ == "__main__":
    main()
