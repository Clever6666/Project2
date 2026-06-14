from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DATA_ROOT = Path(__file__).resolve().parent / "data"

def get_cifar10_loaders(data_dir=None, batch_size=128, val_size=5000, num_workers=4, seed=2026):
    root = Path(data_dir) if data_dir is not None else DATA_ROOT

    # 如果 root 下还没有解压好的文件夹，检查压缩包是否存在，给出提示
    if not (root / "cifar-10-batches-py").exists():
        if (root / "cifar-10-python.tar.gz").exists():
            print("请将压缩包解压到当前目录，或临时设置 download=True 运行一次以便自动解压。")
            print("解压后应得到文件夹：", root / "cifar-10-batches-py")
            raise FileNotFoundError(f"CIFAR-10 数据集未准备好。请确保 {root / 'cifar-10-batches-py'} 存在。")
        else:
            raise FileNotFoundError(
                f"数据集不存在于 {root}。请手动将 cifar-10-python.tar.gz 放入该目录并解压，"
                "或者解压得到 cifar-10-batches-py 文件夹。"
            )

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # download=False：不再尝试联网，使用本地已存在的文件
    train_dataset = datasets.CIFAR10(
        root=root, train=True, download=False, transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        root=root, train=True, download=False, transform=eval_transform
    )
    test_dataset = datasets.CIFAR10(
        root=root, train=False, download=False, transform=eval_transform
    )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
