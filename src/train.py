"""
train.py - Training Loop for Solar Panel Fault Detection Models

This module handles:
1. Training loop with progress bars (tqdm)
2. Validation after each epoch
3. Early stopping to prevent overfitting
4. Learning rate scheduling (ReduceLROnPlateau)
5. Best model checkpoint saving
6. Training history logging (loss + accuracy per epoch)
7. Plotting training curves (loss and accuracy)

Framework: PyTorch
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# Reproducibility
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Trains the model for one complete epoch over the training data.

    Performs forward pass, loss computation, backpropagation, and weight update
    for each batch. Uses tqdm for a visual progress bar.

    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function (CrossEntropyLoss with class weights).
        optimizer (Optimizer): Optimizer (Adam with weight decay).
        device (torch.device): Device to train on (CPU/GPU).

    Returns:
        avg_loss (float): Average training loss for the epoch.
        accuracy (float): Training accuracy for the epoch.
    """
    model.train()  # Set model to training mode (enables dropout, batch norm updates)
    running_loss = 0.0
    correct = 0
    total = 0

    # tqdm progress bar for visual feedback during training
    pbar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in pbar:
        # Move data to device (CPU or GPU)
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass: compute predictions
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass: compute gradients and update weights
        loss.backward()
        optimizer.step()

        # Track statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar with current loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set without gradient computation.

    Used after each training epoch to monitor generalization performance
    and detect overfitting (when val loss increases while train loss decreases).

    Args:
        model (nn.Module): The neural network model.
        val_loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to evaluate on.

    Returns:
        avg_loss (float): Average validation loss.
        accuracy (float): Validation accuracy.
    """
    model.eval()  # Set model to evaluation mode (disables dropout, freezes batch norm)
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for efficiency
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, class_weights, device,
                model_name="model", num_epochs=25, learning_rate=0.001,
                weight_decay=1e-4, patience=5, save_dir="outputs/models",
                plot_dir="outputs/plots"):
    """
    Complete training pipeline with early stopping, LR scheduling, and checkpointing.

    Training Configuration:
    - Loss: CrossEntropyLoss with class weights to handle imbalanced dataset
    - Optimizer: Adam with L2 regularization (weight_decay) to prevent overfitting
    - Scheduler: ReduceLROnPlateau — reduces LR by factor of 0.5 when validation
      loss plateaus for 3 epochs, helping escape local minima
    - Early Stopping: Stops training if validation loss doesn't improve for
      'patience' consecutive epochs, preventing overfitting

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        class_weights (torch.Tensor): Class weights for imbalanced loss.
        device (torch.device): Training device.
        model_name (str): Name for saving files (e.g., "custom_cnn").
        num_epochs (int): Maximum number of training epochs.
        learning_rate (float): Initial learning rate for Adam optimizer.
        weight_decay (float): L2 regularization strength.
        patience (int): Early stopping patience (epochs without improvement).
        save_dir (str): Directory to save model weights.
        plot_dir (str): Directory to save training plots.

    Returns:
        history (dict): Training history with loss and accuracy per epoch.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Move model to device
    model = model.to(device)

    # Loss function with class weights to penalize misclassification of minority classes more
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Adam optimizer with L2 regularization (weight_decay prevents weights from growing too large)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay  # L2 regularization
    )

    # Learning rate scheduler: reduces LR when validation loss stops improving
    # factor=0.5: halve the LR; patience=3: wait 3 epochs before reducing
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Training history for plotting
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': []
    }

    # Early stopping variables
    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_path = os.path.join(save_dir, f"{model_name}_best.pth")

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs} (early stopping patience: {patience})")
    print(f"Learning Rate: {learning_rate}")
    print(f"Weight Decay (L2): {weight_decay}")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}] (LR: {current_lr:.6f})")

        # Training phase
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        # Print epoch results
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # Learning rate scheduling based on validation loss
        scheduler.step(val_loss)

        # Save best model (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, best_model_path)
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.2f}%)")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{patience} epochs")

        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs!")
            print(f"Best Val Acc: {best_val_acc:.2f}% at epoch with Val Loss: {best_val_loss:.4f}")
            break

        print()

    # Plot and save training curves
    plot_training_curves(history, model_name, plot_dir)

    print(f"\nTraining complete for {model_name}!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {best_model_path}")

    return history


def plot_training_curves(history, model_name, plot_dir):
    """
    Plots and saves training/validation loss and accuracy curves.

    These curves are essential for diagnosing model behavior:
    - If train loss decreases but val loss increases → overfitting
    - If both decrease together → good generalization
    - If both are high → underfitting (need more capacity or epochs)

    Args:
        history (dict): Training history with loss and accuracy lists.
        model_name (str): Model name for plot titles and filenames.
        plot_dir (str): Directory to save the plots.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    # ---- Loss Curve ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', markersize=4)
    ax.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss', markersize=4)
    ax.set_title(f'{model_name} - Loss Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(plot_dir, f'{model_name}_loss_curve.png')
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss curve saved to: {loss_path}")

    # ---- Accuracy Curve ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', markersize=4)
    ax.plot(epochs, history['val_acc'], 'r-o', label='Validation Accuracy', markersize=4)
    ax.set_title(f'{model_name} - Accuracy Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    acc_path = os.path.join(plot_dir, f'{model_name}_accuracy_curve.png')
    plt.savefig(acc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Accuracy curve saved to: {acc_path}")


def load_best_model(model, model_path, device):
    """
    Loads the best saved model weights from a checkpoint file.

    Used before evaluation to ensure we test with the best-performing
    model weights (not the last epoch, which may have overfit).

    Args:
        model (nn.Module): Model architecture (must match saved weights).
        model_path (str): Path to the saved checkpoint.
        device (torch.device): Device to load the model onto.

    Returns:
        model (nn.Module): Model with loaded best weights.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Loaded best model from: {model_path}")
    print(f"  Checkpoint epoch: {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.2f}%")
    return model
