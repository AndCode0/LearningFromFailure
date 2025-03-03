import os
import pickle
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
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


def apply_colorization(raw_image, severity, mean_color, attribute_label):
    std_dev = [0.05, 0.02, 0.01, 0.005, 0.002][severity - 1]
    color = torch.clamp(
        mean_color[attribute_label][:, None, None] + torch.randn_like(mean_color[attribute_label][:, None, None]) * std_dev,
        0.0, 1.0
    )
    raw_image = raw_image.to(torch.float32).unsqueeze(0)
    return color * raw_image


COLORED_MNIST_PROTOCOL = {i: lambda img, s: apply_colorization(img, s, fixed_colors, i) for i in range(10)}


def generate_attribute_labels(target_labels, bias_ratio):
    num_classes = target_labels.max().item() + 1
    attr_labels = torch.zeros_like(target_labels)
    for label in range(num_classes):
        indices = (target_labels == label).nonzero(as_tuple=True)[0]
        biased_count = int(len(indices) * bias_ratio)
        unbiased_count = len(indices) - biased_count

        attr_labels[indices[:biased_count]] = label
        for idx in indices[biased_count:]:
            random_label = torch.randint(0, num_classes, (1,))
            while random_label == label:
                random_label = torch.randint(0, num_classes, (1,))
            attr_labels[idx] = random_label

    return attr_labels


def create_colored_mnist(data_dir, skew_ratio, severity, num_workers=2):
    base_dir = os.path.join(data_dir, "ColoredMNIST")
    os.makedirs(base_dir, exist_ok=True)

    attr_names = ["digit", "color"]
    with open(os.path.join(base_dir, "attr_names.pkl"), "wb") as f:
        pickle.dump(attr_names, f)

    for split in ["train", "test"]:
        os.makedirs(os.path.join(base_dir, split), exist_ok=True)
        dataset = datasets.MNIST(data_dir, train=(split == "train"), download=True, transform=transforms.ToTensor())
        bias_ratio = 1. - skew_ratio if split == "train" else 0.1
        color_labels = generate_attribute_labels(torch.tensor(dataset.targets), bias_ratio)

        images, attributes = [], []
        for img, label, color_label in tqdm(zip(dataset.data, dataset.targets, color_labels), total=len(color_labels), leave=False):
            colored_img = COLORED_MNIST_PROTOCOL[color_label.item()](img, severity)
            images.append(np.moveaxis(colored_img.numpy(), 0, 2))
            attributes.append([label.item(), color_label.item()])

        np.save(os.path.join(base_dir, split, "images.npy"), np.array(images, dtype=np.float32))
        np.save(os.path.join(base_dir, split, "attrs.npy"), np.array(attributes, dtype=np.int64))

    train_dataset = TensorDataset(
        torch.tensor(np.load(os.path.join(base_dir, "train", "images.npy"))).permute(0, 3, 1, 2),
        torch.tensor(np.load(os.path.join(base_dir, "train", "attrs.npy"))[:, 0]),
        torch.tensor(np.load(os.path.join(base_dir, "train", "attrs.npy"))[:, 1])
    )
    train_dataset.attr = torch.cat((train_dataset.tensors[1].unsqueeze(1), train_dataset.tensors[2].unsqueeze(1)), dim=1)

    valid_dataset = TensorDataset(
        torch.tensor(np.load(os.path.join(base_dir, "test", "images.npy"))).permute(0, 3, 1, 2),
        torch.tensor(np.load(os.path.join(base_dir, "test", "attrs.npy"))[:, 0]),
        torch.tensor(np.load(os.path.join(base_dir, "test", "attrs.npy"))[:, 1])
    )
    valid_dataset.attr = torch.cat((valid_dataset.tensors[1].unsqueeze(1), valid_dataset.tensors[2].unsqueeze(1)), dim=1)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader
