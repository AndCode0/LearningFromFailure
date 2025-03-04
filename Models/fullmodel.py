import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Union, Generator
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
import wandb

from util import (
    GeneralizedCELoss,
    IdxDataset,
    EMA,
    MultiDimAverageMeter
)
import os


class LfFTrainer:

    def __init__(self, config: dict) -> None:
        """
        Initialize Learning from Failure trainer.
        
        Args:
            config: Configuration object containing training parameters
        """
        self.config = config
        self.device = torch.device(
            config.device
            if hasattr(config, "device")
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Setup data
        self._setup_data()

        # Initialize models
        self._setup_model()

        # Setup loss tracking
        targets = torch.LongTensor(
            self.train_dataset.attr[:, self.config.target_attr_idx]
        )
        self.sample_loss_ema_b = EMA(targets, alpha=self.config.ema_alpha)
        self.sample_loss_ema_d = EMA(targets, alpha=self.config.ema_alpha)

        # define loss
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.bias_criterion = GeneralizedCELoss(q=self.config.gce_q)

    def _setup_data(self) -> None:
        """Setup datasets and data loaders"""
        if self.config.dataset_tag == "CelebA":
            from Data.CelebA import CustomCelebA

            self.num_classes = 2
            transform = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            transform_eval = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

            self.train_dataset = CustomCelebA(
                root=self.config.data_dir,
                split="train",
                target_type="attr",
                transform=transform,
            )
            self.val_dataset = CustomCelebA(
                root=self.config.data_dir,
                split="valid",
                target_type="attr",
                transform=transform_eval,
            )

            # Wrap datasets with IdxDataset for index tracking
            self.train_dataset = IdxDataset(self.train_dataset)
            self.valid_dataset = IdxDataset(self.val_dataset)

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )
            self.val_loader = DataLoader(
                self.valid_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )

        elif self.config["dataset_tag"] == "ColoredMNIST":
            # TODO: add Lorenzo and Francesco's part here
            # import dataset
            # define train and val datasets and loaders
            # self.num_classes = 10 (?)
            
            from Data.c_MNIST import create_colored_mnist

            self.num_classes = 10
            self.train_loader, self.val_loader = create_colored_mnist(
            self.config["data_dir"],
            skew_ratio=self.config["skew_ratio"],
            severity=self.config["severity"],
            num_workers=self.config["num_workers"]
            )
            print("ColoredMNIST dataset loaded successfully.")

            # Wrap datasets with IdxDataset
            self.train_dataset = IdxDataset(self.train_loader.dataset)
            self.valid_dataset = IdxDataset(self.val_loader.dataset)
            
        else:
            raise NotImplementedError

    def _setup_model(self) -> None:
        """Initialize biased and debiased models"""
        if self.config.dataset_tag == "CelebA":
            from torchvision.models import resnet18
            # TODO: load weights from our trained resnet
            model = resnet18(weights=None, num_classes=2).to(self.device)
        elif self.config.dataset_tag == "ColoredMNIST":
            # TODO: define model

            from Models.SimpleConv import SimpleConvNet

            model = SimpleConvNet(num_classes=self.num_classes).to(self.device)
            self.model_b = model.to(self.device)
            self.model_d = model.to(self.device)

         
        else:
            raise ValueError(f"Dataset {self.config.dataset_tag} not supported")

        # Initialize biased and debiased models
        self.model_b = model.to(self.device)
        self.model_d = model.to(self.device)

        if self.config.optimizer_tag == "Adam":
            self.optimizer_b = torch.optim.Adam(
                self.model_b.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            self.optimizer_d = torch.optim.Adam(
                self.model_d.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"Optimizer {self.config.optimizer_tag} not implemented"
            )

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[Dict[str, float], float]:
        """Perform a single training step"""
        index, data, attr = batch
        data = data.to(self.device)
        attr = attr.to(self.device)
        label = attr[:, self.config.target_attr_idx]
        bias_label = attr[:, self.config.bias_attr_idx]

        logit_b = self.model_b(data)
        logit_d = self.model_d(data)

        # Calculate initial losses
        loss_b = self.criterion(logit_b, label).cpu().detach()
        loss_d = self.criterion(logit_d, label).cpu().detach()

        # Store original losses for logging
        loss_per_sample_b = loss_b
        loss_per_sample_d = loss_d

        # Update EMAs
        self.sample_loss_ema_b.update(loss_b, index)
        self.sample_loss_ema_d.update(loss_d, index)

        # Get normalized losses from EMAs
        loss_b = self.sample_loss_ema_b.parameter[index].detach()
        loss_d = self.sample_loss_ema_d.parameter[index].detach()

        # Class-wise normalization
        label_cpu = label.cpu()
        for c in label_cpu.unique():
            class_mask = label_cpu == c
            max_loss_b = self.sample_loss_ema_b.max_loss(c.item())
            max_loss_d = self.sample_loss_ema_d.max_loss(c.item())
            loss_b[class_mask] /= max_loss_b
            loss_d[class_mask] /= max_loss_d

        # Calculate importance weights
        loss_weight = loss_b / (loss_b + loss_d + 1e-8)

        # Final losses for optimization
        loss_b_update = self.bias_criterion(logit_b, label)
        loss_d_update = self.criterion(logit_d, label) * loss_weight.to(self.device)

        loss = loss_b_update.mean() + loss_d_update.mean()

        self.optimizer_b.zero_grad()
        self.optimizer_d.zero_grad()
        loss.backward()
        self.optimizer_b.step()
        self.optimizer_d.step()

        # Prepare metrics for logging
        aligned_mask = label == bias_label
        skewed_mask = ~aligned_mask
        aligned_mask_cpu = aligned_mask.cpu()
        skewed_mask_cpu = skewed_mask.cpu()

        metrics = {
            "loss/b_train": loss_per_sample_b.mean().item(),
            "loss/d_train": loss_per_sample_d.mean().item(),
            "loss_weight/mean": loss_weight.mean().item(),
            "loss_variance/b_ema": self.sample_loss_ema_b.parameter.var().item(),
            "loss_std/b_ema": self.sample_loss_ema_b.parameter.std().item(),
            "loss_variance/d_ema": self.sample_loss_ema_d.parameter.var().item(),
            "loss_std/d_ema": self.sample_loss_ema_d.parameter.std().item(),
        }

        if aligned_mask.any():
            metrics.update(
                {
                    "loss/b_train_aligned": loss_per_sample_b[aligned_mask_cpu]
                    .mean()
                    .item(),
                    "loss/d_train_aligned": loss_per_sample_d[aligned_mask_cpu]
                    .mean()
                    .item(),
                    "loss_weight/aligned": loss_weight[aligned_mask_cpu].mean().item(),
                }
            )

        if skewed_mask.any():
            metrics.update(
                {
                    "loss/b_train_skewed": loss_per_sample_b[skewed_mask_cpu]
                    .mean()
                    .item(),
                    "loss/d_train_skewed": loss_per_sample_d[skewed_mask_cpu]
                    .mean()
                    .item(),
                    "loss_weight/skewed": loss_weight[skewed_mask_cpu].mean().item(),
                }
            )

        # Track number of updated samples (weighted by average importance)
        # TODO: data.size(0) == batch size, no need to calculate it every time. Da testare
        batch_updated = loss_weight.mean().item() * data.size(0)  # self.config.batch_size
        # TODO: log loss_variance/ (b_ema | d_ema)
        return metrics, batch_updated

    def evaluate(self, model: nn.Module) -> Dict[str, Union[float, torch.Tensor]]:
        """Evaluate model on validation set"""
        model.eval()

        # Get dimensions for attribute-wise accuracy tracking
        attr_dims = []
        attr_dims.append(self.num_classes)  # Target attribute dimension

        # Get dimension for bias attribute
        valid_bias_attr = self.valid_dataset.attr[:, self.config.bias_attr_idx]
        attr_dims.append(int(torch.max(valid_bias_attr).item() + 1))

        attrwise_acc_meter = MultiDimAverageMeter(attr_dims)

        with torch.no_grad():
            for index, data, attr in tqdm(
                self.val_loader, leave=False, desc="Validating"
            ):
                data = data.to(self.device)
                attr = attr.to(self.device)
                label = attr[:, self.config.target_attr_idx]

                logit = model(data)
                pred = logit.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).float()

                # Track accuracy across target and bias attributes
                relevant_attrs = attr[
                    :, [self.config.target_attr_idx, self.config.bias_attr_idx]
                ]
                attrwise_acc_meter.add(correct.cpu(), relevant_attrs.cpu())

        accs = attrwise_acc_meter.get_mean()

        # Calculate aligned and skewed accuracies
        eye_tsr = torch.eye(attr_dims[0]).long()

        aligned_acc = accs[eye_tsr == 1].mean().item() * 100
        skewed_acc = accs[eye_tsr == 0].mean().item() * 100
        overall_acc = accs.mean().item() * 100

        model.train()

        return {
            "acc/valid": overall_acc,
            "acc/valid_aligned": aligned_acc,
            "acc/valid_skewed": skewed_acc,
            "attrwise_accs": accs,
        }
    
    def _infinite_loader(self, dataloader: DataLoader) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        """
        Generator that yields batches from a dataloader indefinitely
        
        Args:
            dataloader: DataLoader to iterate over
            
        Yields:
            Batches from the dataloader, restarting when exhausted
        """
        while True:
            for batch in dataloader:
                yield batch

    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        num_updated = 0
        valid_attrwise_accs_list = []

        # Create a infinite data iterator to cycle through batches
        train_gen = self._infinite_loader(self.train_loader)

        for step in tqdm(range(self.config.num_steps)):
            batch = next(train_gen)

            metrics, batch_updated = self.train_step(batch)
            num_updated += batch_updated

            if step % self.config.log_freq == 0:
                wandb.log(metrics, step=step)

            if step % self.config.valid_freq == 0:
                b_metrics = self.evaluate(self.model_b)
                d_metrics = self.evaluate(self.model_d)

                # Store debiased model's attribute-wise accuracies
                valid_attrwise_accs_list.append(d_metrics["attrwise_accs"])

                # Log results
                wandb.log(
                    {
                        "acc/b_valid": b_metrics["acc/valid"],
                        "acc/b_valid_aligned": b_metrics["acc/valid_aligned"],
                        "acc/b_valid_skewed": b_metrics["acc/valid_skewed"],
                        "acc/d_valid": d_metrics["acc/valid"],
                        "acc/d_valid_aligned": d_metrics["acc/valid_aligned"],
                        "acc/d_valid_skewed": d_metrics["acc/valid_skewed"],
                        "num_updated/all": num_updated
                        / self.config.batch_size
                        / self.config.valid_freq,
                    },
                    step=step,
                )

                # Reset counter
                num_updated = 0

        # Save final model
        if self.config.save_model:
            os.makedirs(self.config.save_dir, exist_ok=True)
            model_path = os.path.join(
                self.config.save_dir, f"{wandb.run.name}_debiased_model.pt"
            )
            torch.save(
                {
                    "step": step,
                    "state_dict": self.model_d.state_dict(),
                    "optimizer": self.optimizer_d.state_dict(),
                    "valid_attrwise_accs": (
                        torch.stack(valid_attrwise_accs_list)
                        if valid_attrwise_accs_list
                        else None
                    ),
                },
                model_path,
            )

            wandb.save(model_path)

        return {
            "model_d": self.model_d,
            "valid_attrwise_accs": (
                torch.stack(valid_attrwise_accs_list)
                if valid_attrwise_accs_list
                else None
            ),
        }
