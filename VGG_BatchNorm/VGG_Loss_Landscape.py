import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch import nn
import numpy as np
import random
from tqdm import tqdm

from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader


# ======================
# Device
# ======================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ======================
# Seed
# ======================
def set_random_seeds(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================
# Accuracy
# ======================
def get_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = torch.argmax(out, dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


# ======================
# TRAIN (loss + accuracy + grad cosine)
# ======================
def train(model, optimizer, criterion, train_loader, val_loader, epochs=10):

    model.to(device)

    step_losses = []
    epoch_losses = []
    epoch_accs = []
    grad_cosines = []

    prev_grad = None

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"[{model.__class__.__name__}] Epoch {epoch+1}/{epochs}")

        for x, y in loop:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()

            # ===== gradient flatten (last layer) =====
            grad_vec = []

            for name, p in model.named_parameters():
                if "classifier" in name and p.grad is not None:
                    grad_vec.append(p.grad.view(-1))

            grad_vec = torch.cat(grad_vec)

            # cosine similarity (gradient predictiveness)
            if prev_grad is None:
                cos_sim = 0.0
            else:
                cos_sim = torch.nn.functional.cosine_similarity(
                    grad_vec, prev_grad, dim=0
                ).item()

            prev_grad = grad_vec.detach()

            optimizer.step()

            step_losses.append(loss.item())
            running_loss += loss.item()
            grad_cosines.append(cos_sim)

            loop.set_postfix(loss=loss.item(), grad_cos=f"{cos_sim:.4f}")

        epoch_losses.append(running_loss / len(train_loader))
        epoch_accs.append(get_accuracy(model, val_loader))

        print(f"[Epoch {epoch+1}] Val Acc: {epoch_accs[-1]:.4f}")

    return (
        np.array(step_losses),
        np.array(epoch_losses),
        np.array(epoch_accs),
        np.array(grad_cosines)
    )


# ======================
# LOSS LANDSCAPE
# ======================
def build_landscape(all_runs):

    all_runs = np.array(all_runs, dtype=object)

    steps = len(all_runs[0])

    max_curve, min_curve = [], []

    for i in range(steps):
        vals = [run[i] for run in all_runs]
        max_curve.append(np.max(vals))
        min_curve.append(np.min(vals))

    return np.array(min_curve), np.array(max_curve)


# ======================
# LOSS PLOT
# ======================
def plot_landscape(min_c, max_c, title, path, color):

    x = np.arange(len(min_c))

    plt.figure(figsize=(10, 5))

    plt.plot(x, (min_c + max_c) / 2, color=color, label="mean")
    plt.fill_between(x, min_c, max_c, color=color, alpha=0.25)

    plt.title(title)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()

    plt.savefig(path, dpi=300)
    plt.close()


# ======================
# LR CURVES
# ======================
def plot_lr_curves(all_runs, lrs, title, path):

    plt.figure(figsize=(10,5))

    for i, run in enumerate(all_runs):
        plt.plot(run, label=f"lr={lrs[i]}")

    plt.title(title)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.savefig(path, dpi=300)
    plt.close()


# ======================
# ACCURACY CURVES
# ======================
def plot_accuracy(vgg_acc, bn_acc, lrs):

    plt.figure(figsize=(10,5))

    x = np.arange(len(vgg_acc[0]))

    # ===== VGG (No BN) =====
    for i, acc in enumerate(vgg_acc):
        plt.plot(
            x, acc,
            linestyle="--",
            marker="o",
            linewidth=1.5,
            label=f"No BN | lr={lrs[i]}"
        )

    # ===== BN =====
    for i, acc in enumerate(bn_acc):
        plt.plot(
            x, acc,
            linestyle="-",
            marker="s",
            linewidth=1.8,
            label=f"BN | lr={lrs[i]}"
        )

    plt.title("Validation Accuracy: BN vs No BN (LR Comparison)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)

    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()

    plt.savefig("accuracy_compare.png", dpi=300)
    plt.close()


# ======================
# GRADIENT PREDICTIVENESS
# ======================
def plot_grad_cosine(vgg_cos, bn_cos):

    plt.figure(figsize=(10,4))

    plt.plot(np.concatenate(vgg_cos), label="No BN", color="blue")
    plt.plot(np.concatenate(bn_cos), label="BN", color="red")

    plt.title("Gradient Predictiveness (Cosine Similarity)")
    plt.xlabel("Training Steps")
    plt.ylabel("Cosine Similarity")
    plt.grid()
    plt.legend()

    plt.savefig("grad_cosine_compare.png", dpi=300)
    plt.close()


# ======================
# MAIN
# ======================
if __name__ == "__main__":

    train_loader = get_cifar_loader(train=True)
    val_loader = get_cifar_loader(train=False)

    learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]
    epochs = 30


    # ======================
    # VGG NO BN
    # ======================
    print("\n===== TRAINING VGG_A (NO BN) =====\n")

    vgg_runs = []
    vgg_accs = []
    vgg_cos = []

    for lr in learning_rates:

        print(f"\n[INFO] Training VGG_A | LR={lr}")
        set_random_seeds()

        model = VGG_A().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        step_loss, epoch_loss, acc, cos = train(
            model, optimizer, criterion,
            train_loader, val_loader, epochs
        )

        vgg_runs.append(step_loss)
        vgg_accs.append(acc)
        vgg_cos.append(cos)


    # ======================
    # VGG BN
    # ======================
    print("\n===== TRAINING VGG_A + BN =====\n")

    bn_runs = []
    bn_accs = []
    bn_cos = []

    for lr in learning_rates:

        print(f"\n[INFO] Training VGG_A_BN | LR={lr}")
        set_random_seeds()

        model = VGG_A_BatchNorm().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        step_loss, epoch_loss, acc, cos = train(
            model, optimizer, criterion,
            train_loader, val_loader, epochs
        )

        bn_runs.append(step_loss)
        bn_accs.append(acc)
        bn_cos.append(cos)


    # ======================
    # LOSS LANDSCAPE
    # ======================
    vgg_min, vgg_max = build_landscape(vgg_runs)
    bn_min, bn_max = build_landscape(bn_runs)

    x = np.arange(len(vgg_min))

    plt.figure(figsize=(12,6))

    plt.fill_between(x, vgg_min, vgg_max, alpha=0.25, color='blue', label='VGG No BN')
    plt.fill_between(x, bn_min, bn_max, alpha=0.25, color='red', label='VGG BN')

    plt.title("Loss Landscape Comparison (BN vs No BN)")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()

    plt.savefig("loss_landscape_compare.png", dpi=300)
    plt.close()


    # ======================
    # LR LOSS CURVES
    # ======================
    plot_lr_curves(vgg_runs, learning_rates,
                   "VGG No BN - LR Comparison",
                   "vgg_lr.png")

    plot_lr_curves(bn_runs, learning_rates,
                   "VGG BN - LR Comparison",
                   "bn_lr.png")


    # ======================
    # ACCURACY
    # ======================
    plot_accuracy(vgg_accs, bn_accs, learning_rates)


    # ======================
    # GRADIENT PREDICTIVENESS
    # ======================
    plot_grad_cosine(vgg_cos, bn_cos)

    print("\nDONE. ALL EXPERIMENTS COMPLETED.")