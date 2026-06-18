import csv
import os
import numpy as np


def plot_validation_curves(path, history, model_name):
    """绘制验证损失和准确率随训练轮次变化的曲线图并保存。"""
    # 设置 Matplotlib 后端，避免在无显示器环境下出错
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 从历史记录中提取数据
    epochs = [row["epoch"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    val_acc = [row["val_acc"] for row in history]

    # 创建输出目录
    path.parent.mkdir(parents=True, exist_ok=True)

    # 创建子图：1行2列
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

    # 损失曲线
    ax_loss.plot(epochs, val_loss, marker="o", linewidth=1.5)
    ax_loss.set_title(f"{model_name} validation loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)

    # 准确率曲线
    ax_acc.plot(epochs, val_acc, marker="o", linewidth=1.5)
    ax_acc.set_title(f"{model_name} validation accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(True, alpha=0.3)

    # 调整布局并保存
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def load_history_csv(path):
    history = []
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            history.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "train_acc": float(row["train_acc"]),
                    "val_loss": float(row["val_loss"]),
                    "val_acc": float(row["val_acc"]),
                    "best_val_acc": float(row["best_val_acc"]),
                    "epoch_time_sec": float(row["epoch_time_sec"]),
                    "elapsed_time_sec": float(row["elapsed_time_sec"]),
                }
            )
    return history

def plot_history_comparison(path, history_items, title):
    """
    绘制多个训练历史的验证损失和准确率对比曲线，并保存为图片。

    Args:
        path: 保存路径（Path 对象或字符串）
        history_items: 列表，元素为 (label, history) 元组，
                       history 是包含 'epoch', 'val_loss', 'val_acc' 的字典列表
        title: 图表总标题（用于子图标题）
    """
    # 设置 Matplotlib 后端，避免在没有显示器的环境下出错
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 创建输出目录
    path.parent.mkdir(parents=True, exist_ok=True)

    # 创建子图：1行2列（损失图 + 准确率图）
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    # 遍历每个配置的历史数据，绘制曲线
    for label, history in history_items:
        epochs = [row["epoch"] for row in history]
        val_loss = [row["val_loss"] for row in history]
        val_acc = [row["val_acc"] for row in history]

        ax_loss.plot(epochs, val_loss, linewidth=1.7, label=label)
        ax_acc.plot(epochs, val_acc, linewidth=1.7, label=label)

    # 设置损失子图样式
    ax_loss.set_title(f"{title} validation loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=8)

    # 设置准确率子图样式
    ax_acc.set_title(f"{title} validation accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(fontsize=8)

    # 调整布局并保存
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
import numpy as np  # 确保已导入（函数中使用了 np.arange 和 matrix.max()）


def plot_confusion_matrix(path, matrix, class_names, model_name):
    """
    绘制混淆矩阵的热力图并保存为图片。

    Args:
        path: 保存路径（Path 对象或字符串）
        matrix: 混淆矩阵（2D numpy 数组），形状为 (n_classes, n_classes)
        class_names: 类别名称列表，长度与矩阵维度相同
        model_name: 模型名称，用于图表标题
    """
    # 设置 Matplotlib 后端，避免在没有显示器的环境下出错
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 创建输出目录
    path.parent.mkdir(parents=True, exist_ok=True)

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(8, 7))

    # 绘制混淆矩阵热力图（蓝色系）
    im = ax.imshow(matrix, cmap="Blues")

    # 添加颜色条
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 设置标题和轴标签
    ax.set_title(f"{model_name} test confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # 设置刻度位置和标签
    n_classes = len(class_names)
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # 在每个单元格中添加数值文本
    # 动态选择文字颜色：深色背景用白色，浅色背景用黑色
    threshold = matrix.max() * 0.6 if matrix.size > 0 else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text_color = "white" if value > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=text_color, fontsize=8)

    # 调整布局并保存
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)