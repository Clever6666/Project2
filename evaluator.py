import torch

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return running_loss / total, correct / total

@torch.no_grad()
def compute_confusion_matrix(model, loader, num_classes, device):
    model.eval()
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu()

        for target, prediction in zip(labels.view(-1), predictions.view(-1)):
            matrix[target.long(), prediction.long()] += 1

    return matrix.numpy()