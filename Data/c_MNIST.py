import os
import pickle
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tqdm import tqdm

# Fixed Colors for Each Digit (0-9)
fixed_colors = torch.tensor([
    [1.0, 0.0, 0.0],  # Red for 0
    [0.0, 1.0, 0.0],  # Green for 1
    [0.0, 0.0, 1.0],  # Blue for 2
    [1.0, 1.0, 0.0],  # Yellow for 3
    [1.0, 0.0, 1.0],  # Pink for 4
    [0.0, 1.0, 1.0],  # Cyan for 5
    [0.5, 0.0, 0.5],  # Purple for 6
    [0.5, 0.5, 0.0],  # Olive for 7
    [0.0, 0.5, 0.5],  # Teal for 8
    [0.5, 0.5, 0.5]   # Gray for 9
])


class ColoredMNIST(Dataset):
    def __init__(self, data_dir, split, skew_ratio, severity):
        self.data_dir = data_dir
        self.split = split
        self.severity = severity
        self.dataset = datasets.MNIST(data_dir, train=(split == 'train'), download=True, transform=transforms.ToTensor())
        self.bias_ratio = 1. - skew_ratio if split == 'train' else 0.1
        self.color_labels = self.generate_attribute_labels(torch.tensor(self.dataset.targets))
        self.images, self.attrs = self.apply_colorization()
        self.attr = torch.tensor(self.attrs, dtype=torch.int64)

    def generate_attribute_labels(self, target_labels):
        num_classes = target_labels.max().item() + 1
        attr_labels = torch.zeros_like(target_labels)
        for label in range(num_classes):
            indices = (target_labels == label).nonzero(as_tuple=True)[0]
            biased_count = int(len(indices) * self.bias_ratio)
            attr_labels[indices[:biased_count]] = label
            for idx in indices[biased_count:]:
                random_label = torch.randint(0, num_classes, (1,))
                while random_label == label:
                    random_label = torch.randint(0, num_classes, (1,))
                attr_labels[idx] = random_label
        return attr_labels

    def apply_colorization(self):
        images, attrs = [], []
        std_dev = [0.05, 0.02, 0.01, 0.005, 0.002][self.severity - 1]
        for img, label, color_label in tqdm(zip(self.dataset.data, self.dataset.targets, self.color_labels), total=len(self.color_labels), leave=False):
            color = torch.clamp(
                fixed_colors[color_label][:, None, None] + torch.randn_like(fixed_colors[color_label][:, None, None]) * std_dev,
                0.0, 1.0
            )
            img = img.float().unsqueeze(0)
            colored_img = color * img
            images.append(np.moveaxis(colored_img.numpy(), 0, 2))
            attrs.append([label.item(), color_label.item()])
        return np.array(images, dtype=np.float32), attrs

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx]).permute(2, 0, 1)
        attr = torch.tensor(self.attrs[idx], dtype=torch.int64)  # Convert list to tensor
        return idx, img, attr



    def __len__(self):
        return len(self.images)


def create_colored_mnist(data_dir, skew_ratio, severity, num_workers=2):
    train_dataset = ColoredMNIST(data_dir, 'train', skew_ratio, severity)
    valid_dataset = ColoredMNIST(data_dir, 'test', skew_ratio, severity)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader