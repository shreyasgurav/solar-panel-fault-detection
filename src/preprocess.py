"""
preprocess.py - Data Preprocessing & Augmentation for Solar Panel Fault Detection

This module handles:
1. Loading images from the dataset directory
2. Applying preprocessing (resize, normalize)
3. Data augmentation for training set (to improve generalization)
4. Stratified train/val/test split (to handle class imbalance)
5. Computing class weights (to address imbalanced classes)

Framework: PyTorch with torchvision transforms
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import Counter

# ============================================================
# Reproducibility: Set random seeds for consistent results
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# Constants
# ============================================================
IMG_SIZE = 224  # Standard input size for CNN and transfer learning models (EfficientNet, ResNet)
BATCH_SIZE = 32
NUM_WORKERS = 2

# ImageNet mean and std — used because our transfer learning model (EfficientNetB0)
# was pretrained on ImageNet. Normalizing with these values ensures the input distribution
# matches what the pretrained model expects, leading to better feature extraction.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Class names in the dataset (6 fault categories)
CLASS_NAMES = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']


def get_train_transforms():
    """
    Returns the composition of transforms for the training set.

    Augmentation justifications:
    - RandomHorizontalFlip: Solar panels can appear from either horizontal direction;
      flipping simulates this variation without changing the fault characteristics.
    - RandomVerticalFlip: Panels may be photographed from different orientations;
      vertical flip adds rotational invariance.
    - RandomRotation(20): Slight camera angle variations in real-world captures;
      20 degrees is enough to add variety without distorting panel structure.
    - RandomResizedCrop: Simulates zoom variation and different framing of panels;
      scale 0.8-1.0 keeps most of the panel visible.
    - ColorJitter: Accounts for different lighting conditions, times of day, and
      weather; adjusting brightness/contrast/saturation/hue makes model robust to
      illumination changes.
    - RandomAffine: Adds slight translation and scaling to simulate camera movement
      and different distances from the panel.
    - Normalize with ImageNet stats: Required for transfer learning compatibility.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),  # Resize slightly larger for cropping
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),  # Random zoom/crop
        transforms.RandomHorizontalFlip(p=0.5),  # 50% chance horizontal flip
        transforms.RandomVerticalFlip(p=0.3),  # 30% chance vertical flip
        transforms.RandomRotation(degrees=20),  # Rotate up to ±20 degrees
        transforms.ColorJitter(
            brightness=0.2,  # ±20% brightness change
            contrast=0.2,    # ±20% contrast change
            saturation=0.2,  # ±20% saturation change
            hue=0.1          # ±10% hue shift
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  # Up to 10% translation
            scale=(0.9, 1.1)       # ±10% scale
        ),
        transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)  # ImageNet normalization
    ])


def get_val_test_transforms():
    """
    Returns the composition of transforms for validation and test sets.

    No augmentation is applied — only deterministic preprocessing to ensure
    consistent and fair evaluation. We resize and center crop to get a clean
    224x224 input, then normalize with ImageNet statistics.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize to exact size
        transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)  # ImageNet normalization
    ])


def get_display_transforms():
    """
    Returns transforms for displaying images (no normalization).
    Used for visualizing augmented samples in the notebook.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])


class SolarPanelDataset(Dataset):
    """
    Custom PyTorch Dataset for Solar Panel Fault Detection.

    Loads images from a list of file paths with their corresponding labels.
    Applies the specified transforms for preprocessing and augmentation.
    Handles corrupt/unreadable images gracefully by returning a random valid sample.
    """

    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): List of absolute paths to image files.
            labels (list): List of integer labels corresponding to each image.
            transform (callable, optional): Transform to apply to each image.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Loads and returns one sample (image, label) at the given index.
        Converts images to RGB to handle grayscale or RGBA images uniformly.
        """
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            # If an image is corrupt, return a random valid sample instead
            print(f"Warning: Could not load image {self.image_paths[idx]}: {e}")
            random_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(random_idx)


def load_dataset(data_dir):
    """
    Loads all image paths and their labels from the dataset directory.

    The dataset is organized as:
        data_dir/
            Bird-drop/
            Clean/
            Dusty/
            Electrical-damage/
            Physical-Damage/
            Snow-Covered/

    Each subfolder name corresponds to a class label.

    Args:
        data_dir (str): Path to the root dataset directory.

    Returns:
        image_paths (list): List of absolute file paths to all images.
        labels (list): List of integer labels (0-5) for each image.
        class_names (list): List of class name strings.
    """
    image_paths = []
    labels = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Directory not found for class '{class_name}': {class_dir}")
            continue

        for img_name in os.listdir(class_dir):
            ext = os.path.splitext(img_name)[1].lower()
            if ext in valid_extensions:
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(class_idx)

    print(f"Loaded {len(image_paths)} images across {len(CLASS_NAMES)} classes")
    return image_paths, labels, CLASS_NAMES


def split_dataset(image_paths, labels, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Splits the dataset into train, validation, and test sets using stratified splitting.

    Stratified split ensures each split has the same proportion of each class as the
    full dataset — critical when dealing with class imbalance (e.g., Physical-Damage
    has fewer images than Bird-drop).

    Args:
        image_paths (list): All image file paths.
        labels (list): Corresponding integer labels.
        train_ratio (float): Proportion for training (default 0.70).
        val_ratio (float): Proportion for validation (default 0.15).
        test_ratio (float): Proportion for testing (default 0.15).

    Returns:
        Dictionary with train/val/test paths and labels.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # First split: separate test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels,
        test_size=test_ratio,
        stratify=labels,  # Stratified to maintain class distribution
        random_state=SEED
    )

    # Second split: separate train and validation from the remaining data
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=relative_val_ratio,
        stratify=train_val_labels,  # Stratified again
        random_state=SEED
    )

    # Print split statistics
    print(f"\nDataset Split:")
    print(f"  Training:   {len(train_paths)} images ({train_ratio*100:.0f}%)")
    print(f"  Validation: {len(val_paths)} images ({val_ratio*100:.0f}%)")
    print(f"  Test:       {len(test_paths)} images ({test_ratio*100:.0f}%)")

    # Print class distribution in each split
    for split_name, split_labels in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
        counter = Counter(split_labels)
        dist = {CLASS_NAMES[k]: v for k, v in sorted(counter.items())}
        print(f"  {split_name} distribution: {dist}")

    return {
        'train': (train_paths, train_labels),
        'val': (val_paths, val_labels),
        'test': (test_paths, test_labels)
    }


def compute_class_weights(labels):
    """
    Computes class weights inversely proportional to class frequency.

    This addresses class imbalance by giving higher weight to underrepresented
    classes (e.g., Physical-Damage with only ~69 images) during loss computation.
    The model is penalized more for misclassifying minority class samples.

    Formula: weight_i = total_samples / (num_classes * count_i)

    Args:
        labels (list): List of integer labels.

    Returns:
        torch.FloatTensor: Class weights tensor of shape (num_classes,).
    """
    counter = Counter(labels)
    total = len(labels)
    num_classes = len(CLASS_NAMES)

    weights = []
    for i in range(num_classes):
        count = counter.get(i, 1)  # Avoid division by zero
        weight = total / (num_classes * count)
        weights.append(weight)

    class_weights = torch.FloatTensor(weights)
    print(f"\nClass Weights (higher = more underrepresented):")
    for name, w in zip(CLASS_NAMES, weights):
        print(f"  {name}: {w:.4f}")

    return class_weights


def get_weighted_sampler(labels):
    """
    Creates a WeightedRandomSampler for the training DataLoader.

    This oversamples minority classes during training so that each batch
    has a more balanced class distribution. Combined with class weights
    in the loss function, this provides a two-pronged approach to handling
    class imbalance.

    Args:
        labels (list): List of integer labels for the training set.

    Returns:
        WeightedRandomSampler: Sampler that oversamples minority classes.
    """
    counter = Counter(labels)
    # Weight for each sample = 1 / (frequency of its class)
    sample_weights = [1.0 / counter[label] for label in labels]
    sample_weights = torch.FloatTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # Allow resampling of minority class images
    )
    return sampler


def create_dataloaders(data_dir, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Main function to create train, validation, and test DataLoaders.

    Pipeline:
    1. Load all images and labels from disk
    2. Split into train/val/test (70/15/15, stratified)
    3. Apply augmentation transforms to training set only
    4. Create weighted sampler for training to handle class imbalance
    5. Return DataLoaders ready for model training/evaluation

    Args:
        data_dir (str): Path to the dataset root directory.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of parallel data loading workers.

    Returns:
        train_loader, val_loader, test_loader: PyTorch DataLoaders.
        class_weights: Tensor of class weights for the loss function.
        class_names: List of class name strings.
    """
    # Step 1: Load dataset
    image_paths, labels, class_names = load_dataset(data_dir)

    # Step 2: Stratified split
    splits = split_dataset(image_paths, labels)

    # Step 3: Compute class weights from training labels
    train_paths, train_labels = splits['train']
    val_paths, val_labels = splits['val']
    test_paths, test_labels = splits['test']
    class_weights = compute_class_weights(train_labels)

    # Step 4: Create datasets with appropriate transforms
    train_dataset = SolarPanelDataset(train_paths, train_labels, transform=get_train_transforms())
    val_dataset = SolarPanelDataset(val_paths, val_labels, transform=get_val_test_transforms())
    test_dataset = SolarPanelDataset(test_paths, test_labels, transform=get_val_test_transforms())

    # Step 5: Create weighted sampler for training data
    train_sampler = get_weighted_sampler(train_labels)

    # Step 6: Create DataLoaders
    # Note: When using a sampler, shuffle must be False (sampler handles ordering)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,  # Speeds up CPU-to-GPU transfer
        drop_last=True    # Drop incomplete last batch for consistent batch norm
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\nDataLoaders created (batch_size={batch_size}):")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader, class_weights, class_names
