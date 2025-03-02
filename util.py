import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneralizedCELoss(nn.Module):
    def __init__(self, q=0.7):
        super().__init__()
        self.q = q

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_prob = probs[torch.arange(probs.shape[0]), targets]
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        return ce_loss * torch.pow(targets_prob.detach(), self.q)
    
class IdxDataset(torch.utils.data.Dataset):
    """Dataset wrapper that also returns indices"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.attr = getattr(dataset, 'attr', None)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        return idx, img, target
        
# TODO: pass num_classes as argument, minimal benefit
class EMA:
    def __init__(self, targets: torch.Tensor, alpha: float = 0.7):
        self.alpha = alpha
        self.parameter = torch.zeros(targets.shape[0], dtype=torch.float)
        self.num_classes = int(targets.max().item() + 1)

        # Create class masks for faster lookup
        self.class_masks = []
        for c in range(self.num_classes):
            self.class_masks.append((targets == c))

    def update(self, loss: torch.Tensor, indices: torch.Tensor):
        self.parameter[indices] = self.alpha * self.parameter[indices] + (1 - self.alpha) * loss

    def max_loss(self, label: int) -> float:
        class_mask = self.class_masks[label]
        class_losses = self.parameter[class_mask]

        # Handle empty case and return max
        if len(class_losses) == 0:
            return 1.0

        max_val = class_losses.max().item()
        return max_val if max_val > 0 else 1.0

class MultiDimAverageMeter:
    def __init__(self, dims: list):
        self.dims = dims
        self.reset()

    def reset(self):
        self.counts = torch.zeros(self.dims)
        self.sums = torch.zeros(self.dims)

    def add(self, values: torch.Tensor, indices: torch.Tensor):
        # Add values at specified indices"""
        indices_tuple = tuple(indices[:, i] for i in range(indices.shape[1]))
        self.total[indices_tuple] += values
        self.count[indices_tuple] += 1

    def get_mean(self) -> torch.Tensor:
        # Return mean values across all dimensions
        return self.sums / (self.counts + 1e-8)