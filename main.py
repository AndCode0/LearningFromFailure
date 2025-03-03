import wandb
import torch
import Models.fullmodel as lff
from typing import Dict, Any

conf: Dict[str, Any] = {
    # Dataset configuration
    "dataset_tag": "ColoredMNIST",                 # Dataset name (CelebA or ColoredMNIST)
    "data_dir": "./Data",                    # Root directory for dataset storage
    "target_attr_idx": 1,                    # 9 BlondHair attribute index for CelebA, 1 label for cMNIST
    "bias_attr_idx": 1,                     # 20 Male attribute index for CelebA, 

    # Model configuration
    "model_tag": "SimpleConvNet",                 # Model architecture to use
    
    # Training hyperparameters
    "batch_size": 256,                       # Batch size for training and validation
    "num_steps": 10000,                      # Total number of training steps | original: (636 * 200)
    "num_workers": 0,                        # Number of worker processes for data loading
    
    # Optimizer settings
    "optimizer_tag": "Adam",                 # Optimizer type
    "learning_rate": 0.001,                  # Learning rate for optimizer
    "weight_decay": 1e-4,                    # Weight decay for regularization
    
    # LfF-specific parameters
    "gce_q": 0.7,                            # q parameter for Generalized Cross Entropy
    "ema_alpha": 0.9,                        # Alpha parameter for EMA loss tracking
    
    # Logging and evaluation
    "valid_freq": 100,                       # Validate every N steps
    "log_freq": 10,                          # Log metrics every N steps
    "save_model": True,                      # Whether to save the model at the end
    "save_dir": "./Results",   # Directory to save model weights
    
    # Hardware settings
    "device": "cuda" if torch.cuda.is_available() else "cpu", # Device to use for training
}

wandb.login(key='cf0fbaf900fabe0767c84a925064c7e6232d11e6')
#wandb.init(mode="offline", config=conf)

with wandb.init(
     entity="learning-from-failure", project="fullModel", id="LorenzoTry1", config=conf
 ) as run:
    model = lff.LfFTrainer(wandb.config)
    results = model.train()  

wandb.finish()