import wandb
import torch
import Models.fullmodel as lff
from util import set_seed
from typing import Dict, Any
from util import set_seed
import os

config: Dict[str, Any] = dict(
    # Dataset configuration
    dataset_tag = "CelebA",              # CelebA | ColoredMNIST
    data_dir = "INSERT/PATH/TO/DATA_DIR",# Root directory for dataset storage
    target_attr_idx = 9,                 # BlondHair (CelebA): 9 | label (cMNST): 0 
    bias_attr_idx = 20,                  # Male (CelebA): 20 | color (cMNST): 1 
    skew_ratio = 0.02,
    severity = 2,
    # Model configuration
    model_tag = "ResNet18",              # Model architecture:  ResNet18 | SimpleConvNet
    weights = None,                      # 'PATH/TO/WEIGHTS' | None | pretrained

    # Training hyperparameters
    batch_size = 256,                     # Batch size for training and validation
    num_steps = 10000,                    # Total number of training steps | original: (636 * 200)
    num_workers = 0,                      # Number of worker processes for data loading

    # Optimizer settings
    optimizer_tag = "Adam",               # Optimizer type
    learning_rate = 0.001,                # Learning rate for optimizer
    weight_decay = 1e-4,                  # Weight decay for regularization

    # LfF-specific parameters
    gce_q = 0.7,                          # q parameter for Generalized Cross Entropy
    ema_alpha = 0.9,                      # Alpha parameter for EMA loss tracking

    # Logging and evaluation
    valid_freq = 100,                     # Validate every N steps
    log_freq = 10,                        # Log metrics every N steps
    save_model = True,                    # Whether to save the model at the end
    save_dir = "INSERT/PATH/TO/SAVE_DIR", # Directory to save model weights

    # Hardware settings
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ),  # Device to use for training
)
if config['momentum'] is not None:
    raise ValueError(f'Optimizer {config["optimizer_tag"]} doesn\'t require a momentum. Set it to None, or deal with the consequences')

set_seed(42)

wandb.login(key="INSERISCI-KEY")

with wandb.init(
    entity="learning-from-failure",
    project="fullModel",
    id="NewRun",
    conf=config,
    name = "NameFileOnceSaved",
    mode="offline",                         # "online" | "offline" | "disabled"
):
    model = lff.LfFTrainer(wandb.config)
    try:
        results = model.train()
    except KeyboardInterrupt:
        # Save final model
        if wandb.config.save_model:
            os.makedirs(wandb.config.save_dir, exist_ok=True)
            model_path = os.path.join(
                wandb.config.save_dir, f"{wandb.run.name}__models.pt"
            )

            torch.save(
                {
                    # debiased model
                    "state_dict_d": model.model_d.state_dict(),
                    "optimizer_d": model.optimizer_d.state_dict(),
                    # biased model
                    "state_dict_b": model.model_b.state_dict(),
                    "optimizer_b": model.optimizer_b.state_dict(),
                },
                model_path,
            ) 
    finally:
            wandb.finish()
