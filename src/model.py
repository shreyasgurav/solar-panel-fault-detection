"""
model.py - Model Architectures for Solar Panel Fault Detection

This module defines two models for comparison:
    Model A: Custom CNN built from scratch (4 Conv blocks + FC layers)
    Model B: Transfer Learning with EfficientNetB0 (pretrained on ImageNet)

Both models classify solar panel images into 6 fault categories.

Framework: PyTorch
"""

import os
import ssl
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# ============================================================
# Fix SSL certificate issue on macOS (needed to download pretrained weights)
# ============================================================
if not os.environ.get('SSL_CERT_FILE'):
    try:
        import certifi
        os.environ['SSL_CERT_FILE'] = certifi.where()
        ssl._create_default_https_context = ssl._create_unverified_context
    except ImportError:
        ssl._create_default_https_context = ssl._create_unverified_context

# ============================================================
# Reproducibility
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

NUM_CLASSES = 6  # Bird-drop, Clean, Dusty, Electrical-damage, Physical-Damage, Snow-Covered


class CustomCNN(nn.Module):
    """
    Model A — Custom CNN Architecture for Solar Panel Fault Detection.

    Architecture Design Rationale:
    - 4 Convolutional blocks with increasing filter sizes (32→64→128→256)
      to capture features at multiple scales: low-level edges → high-level patterns.
    - Each block: Conv2D → BatchNorm → ReLU → MaxPool2D
        * Conv2D: Extracts spatial features (edges, textures, patterns)
        * BatchNorm: Stabilizes training by normalizing activations, allows higher
          learning rates, and acts as mild regularization
        * ReLU: Non-linear activation that enables learning complex patterns;
          chosen over sigmoid/tanh for faster training (no vanishing gradient)
        * MaxPool2D: Reduces spatial dimensions by 2x, providing translation
          invariance and reducing computation
    - Global Average Pooling: Reduces each feature map to a single value,
      much fewer parameters than Flatten (reduces overfitting on small datasets)
    - Fully connected layers with Dropout (0.5): Final classification with
      strong regularization to prevent overfitting on our ~876 image dataset
    - Softmax output: Produces probability distribution over 6 classes

    Input: 224x224x3 RGB images
    Output: 6-class probability distribution
    """

    def __init__(self, num_classes=NUM_CLASSES):
        """
        Initialize the Custom CNN with 4 convolutional blocks and FC classifier.

        Args:
            num_classes (int): Number of output classes (default: 6).
        """
        super(CustomCNN, self).__init__()

        # ---- Convolutional Block 1 ----
        # Input: 3x224x224 → Output: 32x112x112
        # First block uses 32 filters to detect basic features: edges, corners, color blobs
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ---- Convolutional Block 2 ----
        # Input: 32x112x112 → Output: 64x56x56
        # 64 filters to combine low-level features into textures and simple patterns
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ---- Convolutional Block 3 ----
        # Input: 64x56x56 → Output: 128x28x28
        # 128 filters to detect more complex patterns (crack shapes, dust patterns, snow textures)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ---- Convolutional Block 4 ----
        # Input: 128x28x28 → Output: 256x14x14
        # 256 filters for high-level semantic features (fault type-specific patterns)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ---- Global Average Pooling ----
        # Input: 256x14x14 → Output: 256x1x1
        # Reduces each feature map to a single value; acts as structural regularizer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ---- Classifier Head ----
        # Fully connected layers with Dropout for final classification
        self.classifier = nn.Sequential(
            nn.Flatten(),               # 256x1x1 → 256
            nn.Linear(256, 512),        # Dense layer to learn class-specific combinations
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),            # 50% dropout — strong regularization for small dataset
            nn.Linear(512, 256),        # Second dense layer for further abstraction
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),            # Another dropout layer
            nn.Linear(256, num_classes) # Output layer — raw logits (CrossEntropyLoss applies softmax)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input batch of images, shape (B, 3, 224, 224).

        Returns:
            torch.Tensor: Raw logits of shape (B, num_classes).
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x


class EfficientNetTransfer(nn.Module):
    """
    Model B — Transfer Learning with EfficientNetB0 for Solar Panel Fault Detection.

    Transfer Learning Rationale:
    - EfficientNetB0 was pretrained on ImageNet (1.2M images, 1000 classes), so it
      has already learned rich, general-purpose visual features (edges, textures,
      shapes, objects). These features transfer well to our solar panel task.
    - With only ~876 images, training from scratch is prone to overfitting.
      Transfer learning leverages pretrained knowledge to achieve higher accuracy
      with less data.
    - Strategy:
        Phase 1 (Feature Extraction): Freeze all base layers, train only the custom
        classifier head. This avoids destroying pretrained features.
        Phase 2 (Fine-tuning): Unfreeze the last few layers of EfficientNet and
        train with a smaller learning rate to adapt high-level features to our task.

    Custom Head:
    - GlobalAveragePooling2D: Already in EfficientNet, reduces spatial dimensions
    - Dense(256) + ReLU: Learns task-specific feature combinations
    - Dropout(0.4): Regularization (slightly less than custom CNN since pretrained
      features are more robust)
    - Dense(6): Output logits for 6 classes

    Input: 224x224x3 RGB images (same as EfficientNet's expected input)
    Output: 6-class probability distribution
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        """
        Initialize EfficientNetB0 with a custom classification head.

        Args:
            num_classes (int): Number of output classes (default: 6).
            pretrained (bool): Whether to load ImageNet pretrained weights.
        """
        super(EfficientNetTransfer, self).__init__()

        # Load pretrained EfficientNetB0
        # EfficientNet achieves state-of-the-art accuracy with fewer parameters
        # than ResNet, making it ideal for our relatively small dataset
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.base_model = models.efficientnet_b0(weights=weights)
        else:
            self.base_model = models.efficientnet_b0(weights=None)

        # Get the number of features from EfficientNet's classifier
        num_features = self.base_model.classifier[1].in_features  # 1280 for EfficientNetB0

        # Replace the original classifier with our custom head
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.4),                # Regularization before dense layers
            nn.Linear(num_features, 256),   # Reduce from 1280 to 256 features
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),                # Additional regularization
            nn.Linear(256, num_classes)     # Final classification layer
        )

        # Initially freeze all base layers (Phase 1: Feature Extraction)
        self.freeze_base()

    def freeze_base(self):
        """
        Freeze all layers in the EfficientNet backbone.

        During Phase 1 (feature extraction), we only train the classifier head.
        This prevents the pretrained weights from being destroyed by large gradients
        from random classifier weights.
        """
        for param in self.base_model.features.parameters():
            param.requires_grad = False
        print("Base model layers FROZEN (feature extraction mode)")

    def unfreeze_top_layers(self, num_layers=3):
        """
        Unfreeze the last N layers of EfficientNet for fine-tuning (Phase 2).

        Fine-tuning the top layers allows the model to adapt high-level features
        (shapes, patterns) to our specific solar panel fault detection task,
        while keeping low-level features (edges, textures) fixed.

        Args:
            num_layers (int): Number of top feature blocks to unfreeze.
        """
        # EfficientNetB0 has 9 feature blocks (0-8)
        total_blocks = len(self.base_model.features)
        unfreeze_from = total_blocks - num_layers

        for idx, block in enumerate(self.base_model.features):
            if idx >= unfreeze_from:
                for param in block.parameters():
                    param.requires_grad = True

        print(f"Unfroze last {num_layers} blocks (fine-tuning mode)")

    def forward(self, x):
        """
        Forward pass through EfficientNetB0 + custom classifier.

        Args:
            x (torch.Tensor): Input batch of images, shape (B, 3, 224, 224).

        Returns:
            torch.Tensor: Raw logits of shape (B, num_classes).
        """
        return self.base_model(x)


def get_model_summary(model, input_size=(3, 224, 224), device='cpu'):
    """
    Prints a detailed summary of the model architecture.

    Uses torchsummary to display layer-by-layer information including:
    - Layer type and configuration
    - Output shape at each layer
    - Number of parameters (trainable and non-trainable)

    Args:
        model (nn.Module): The PyTorch model to summarize.
        input_size (tuple): Input tensor dimensions (C, H, W).
        device (str): Device to use for summary computation.
    """
    try:
        from torchsummary import summary
        model = model.to(device)
        print("\n" + "=" * 60)
        print(f"Model: {model.__class__.__name__}")
        print("=" * 60)
        summary(model, input_size=input_size, device=device)
    except ImportError:
        print("torchsummary not installed. Printing basic model structure:")
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


def count_parameters(model):
    """
    Counts and returns the total and trainable parameters in the model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        tuple: (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
