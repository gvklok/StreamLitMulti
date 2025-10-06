import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn import metrics
import csv


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        # Conv1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # max pooling

        # Conv2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Conv3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_one_epoch(model, device, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, device, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def plot_history(history, out_dir='results'):
    os.makedirs(out_dir, exist_ok=True)
    epochs = np.arange(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history['train_loss'], label='train_loss')
    plt.plot(epochs, history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    plt.savefig(os.path.join(out_dir, 'loss_pytorch.png'))
    print('Saved', os.path.join(out_dir, 'loss_pytorch.png'))

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history['train_acc'], label='train_acc')
    plt.plot(epochs, history['val_acc'], label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig(os.path.join(out_dir, 'accuracy_pytorch.png'))
    print('Saved', os.path.join(out_dir, 'accuracy_pytorch.png'))


def main():
    device = get_device()
    print('Using device:', device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load dataset
    dataset = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

    # Split train into train/val
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    batch_size = 64
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    model = SimpleCNN(in_channels=1, num_classes=10).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 50
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    out_dir = 'results'
    os.makedirs(out_dir, exist_ok=True)

    best_val_loss = float('inf')
    best_path = os.path.join(out_dir, 'best_fashion_mnist_cnn_pytorch.pt')

    start = time.time()
    try:
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(model, device, train_loader, criterion, optimizer)
            val_loss, val_acc = evaluate(model, device, val_loader, criterion)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch:02d}/{epochs} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} - val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

            # Save best checkpoint by validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_path)
                print(f"Saved best model (val_loss improved to {best_val_loss:.4f}) -> {best_path}")

    except KeyboardInterrupt:
        print('\nTraining interrupted by user (KeyboardInterrupt). Saving current model and stats...')

    dur = time.time() - start
    print(f"Training finished (or interrupted) in {dur/60:.2f} minutes")

    # Load best model for evaluation if available
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print('Loaded best model from', best_path)

    test_loss, test_acc = evaluate(model, device, test_loader, criterion)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    # Save final model and history
    final_model_path = os.path.join(out_dir, 'fashion_mnist_cnn_pytorch_final.pt')
    torch.save(model.state_dict(), final_model_path)
    print('Saved final model to', final_model_path)

    # Save history to CSV
    history_csv = os.path.join(out_dir, 'history.csv')
    with open(history_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
        for i in range(len(history['train_loss'])):
            writer.writerow([i + 1, history['train_loss'][i], history['train_acc'][i], history['val_loss'][i], history['val_acc'][i]])
    print('Saved training history to', history_csv)

    plot_history(history, out_dir=out_dir)

    # Confusion matrix and classification report on test set
    # Collect predictions
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(targets.numpy().tolist())

    cm = metrics.confusion_matrix(y_true, y_pred)
    report = metrics.classification_report(y_true, y_pred, digits=4)
    print('\nClassification report:\n', report)

    # Save classification report
    report_path = os.path.join(out_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print('Saved classification report to', report_path)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
    cm_path = os.path.join(out_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(cm_path)
    print('Saved confusion matrix to', cm_path)


if __name__ == '__main__':
    main()
