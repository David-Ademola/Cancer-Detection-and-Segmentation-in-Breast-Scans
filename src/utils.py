import gc
import os
import random

import numpy as np
import pandas as pd
import torch
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from monai.metrics.meandice import DiceMetric
from monai.metrics.meaniou import MeanIoU
from scipy.ndimage import gaussian_gradient_magnitude
from skimage import io
from skimage.color import rgb2gray
from skimage.restoration import denoise_tv_chambolle
from skimage.util import img_as_float
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.nn import CrossEntropyLoss, Module
from torch.nn.functional import one_hot
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as func
from tqdm import tqdm

NORMAL, MALIGNANT = 0, 1
SEG_WEIGHT, CLS_WEIGHT = 0.5, 0.5
SIGMA: list[float] = [1.0, 2.0]
LAMBDA_BG: float = 0.05


class JointTransform:
    """
    Applies the same transformations to image and mask pairs.

    Transformations include: random rotation, random flips, and random contrast (image only)
    """

    def __init__(self) -> None:
        super().__init__()
        return

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Random rotation
        angle = random.uniform(-45, 45)
        image = func.rotate(image, angle)
        mask = func.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        # Random flips
        if random.random() < 0.5:
            image, mask = func.hflip(image), func.hflip(mask)
        if random.random() < 0.5:
            image, mask = func.vflip(image), func.vflip(mask)

        # Random contrast (image only)
        if random.random() < 0.5:
            image = func.adjust_contrast(image, random.uniform(0.8, 1.3))

        return image, mask


class BreastDataset(Dataset):
    """
    Represents a dataset for breast images enabling data loading and processing.

    This dataset is designed to preprocess, convert and transform inputs
    (images, masks and labels) into a format usable for machine learning
    models. Images are preprocessed and normalized, masks are processed to
    handle transparency and converted into float format, and both images
    and masks are resized to a standard size. Optionally, transformations
    can be applied for additional augmentation or data preparation
    requirements.

    :ivar items: List of tuples, where each tuple represents an image path,
        mask path and label.
    :type items: list[tuple]
    :ivar image_size: Size of the image after preprocessing and resizing.
    :type image_size: list[int]
    :ivar transforms: Optional transformation pipeline to apply to the image
        and mask.
    :type transforms: JointTransform or None
    """

    def __init__(
        self, images_df: pd.DataFrame, image_size: list[int], transforms: None | JointTransform
    ) -> None:
        super().__init__()

        self.items = [tuple(row) for row in images_df.to_numpy()]
        self.image_size = image_size
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        # Load the image, mask, and label from the dataset
        image_path, mask_path, type_label, presence_label = self.items[index]

        # Apply preprocessing to the image
        image = preprocess_image(image_path, sigma=SIGMA, weight=0.04, denoise=True)

        # Load the mask
        mask = io.imread(mask_path)

        # Check if the mask is transparent
        if mask.ndim == 3 and mask.shape[2] == 4:
            mask = mask[..., 3]
        elif mask.ndim == 3:
            mask = rgb2gray(mask)

        # Convert the mask to float
        mask = img_as_float(mask)

        # Convert the image and mask to a tensor
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)

        # Resize the image and mask to the specified size
        image = func.resize(image, self.image_size)
        mask = func.resize(mask, self.image_size, interpolation=InterpolationMode.NEAREST)

        # Apply transformations if any
        if self.transforms:
            image, mask = self.transforms(image, mask)

        # Convert the labels to a tensor
        type_label = torch.tensor(type_label, dtype=torch.long)
        presence_label = torch.tensor(presence_label, dtype=torch.long)

        return image, (mask, presence_label, type_label)


class BreastTestDataset(Dataset):
    """
    Represents a dataset for Breast Test images, enabling data loading and processing.

    This class extends the `Dataset` class and is specifically designed to handle a
    collection of breast test images. It facilitates accessing, preprocessing, and
    resizing images stored at given file paths. The dataset can be used to loop through
    images, making it compatible with PyTorch data loaders.

    :ivar items: Array containing paths to images in the dataset.
    :type items: list[tuple]
    :ivar image_size: Desired size of the output images after preprocessing and resizing.
    :type image_size: list[int]
    """

    def __init__(self, images_df: pd.DataFrame, image_size: list[int]) -> None:
        super().__init__()

        self.items = [tuple(row) for row in images_df.to_numpy()]
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple:
        image_path, mask_path, label = self.items[index]
        # Preprocess image
        image = preprocess_image(image_path, sigma=SIGMA, weight=0.04, denoise=True)

        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32)
        original_image_size = tuple(image.shape[-2:])

        # Resize
        image = func.resize(image, self.image_size)

        return image, image_path, mask_path, original_image_size, label


# class IoULoss(Module):
#     """
#     Computes the Intersection over Union (IoU) loss, also known as the Jacquard loss.

#     This loss function is particularly useful for semantic segmentation tasks. It measures the
#     dissimilarity between the predicted segmentation map and the ground truth mask by calculating
#     `1 - IoU`. The IoU metric itself is the ratio of the area of intersection to the area of
#     union between the predicted and true masks.

#     The implementation provides flexibility to handle raw logits (by applying softmax),
#     convert integer-based ground truth masks to one-hot encoding, and optionally exclude
#     the background class from the calculation.

#     :ivar include_background: If False, the background class (channel 0) is excluded from the
#         loss calculation. Defaults to True.
#     :type include_background: bool
#     :ivar to_onehot_y: If True, converts the ground truth mask `y_true` from an integer format
#         (B, 1, H, W) to a one-hot encoded format (B, C, H, W). Defaults to True.
#     :type to_onehot_y: bool
#     :ivar softmax: If True, a softmax activation is applied to the prediction `y_pred` tensor
#         along the channel dimension to convert logits into probabilities. Defaults to True.
#     :type softmax: bool
#     :ivar smooth: A small epsilon value added to the numerator and denominator to prevent
#         division by zero and improve numerical stability. Defaults to 1e-8.
#     :type smooth: float
#     """
#     def __init__(
#             self, include_background: bool = True, to_onehot_y: bool = True,
#             softmax: bool = True, smooth: float = 1e-8
#     ) -> None:
#         super().__init__()
#         self.include_background = include_background
#         self.to_onehot_y = to_onehot_y
#         self.softmax = softmax
#         self.smooth = smooth

#     def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
#         """
#         Calculates the IoU loss for a batch of predictions and targets.

#         :param y_pred: The model's raw output logits. The expected shape is (B, C, H, W), where B is
#             the batch size, C is the number of classes, and H, W are the height and width.
#         :type y_pred: torch.Tensor
#         :param y_true: The ground truth labels. The expected shape is (B, 1, H, W) with integer
#             class indices if `to_onehot_y` is True, or (B, C, H, W) in one-hot format if False.
#         :type y_true: torch.Tensor
#         :return: A scalar tensor representing the mean IoU loss for the batch.
#         :rtype: torch.Tensor
#         """
#         # Convert logits to probabilities
#         if self.softmax:
#             y_pred = torch.softmax(y_pred, dim=1)

#         # Convert ground truth to one-hot format
#         if self.to_onehot_y:
#             y_true = one_hot(y_true.squeeze(1).long(), num_classes=y_pred.shape[1])
#             y_true = y_true.permute(0, 3, 1, 2).float()

#         # Exclude background
#         if not self.include_background:
#             y_pred = y_pred[:, 1:]
#             y_true = y_true[:, 1:]

#         # Flatten spatial dimensions
#         y_pred = y_pred.contiguous().view(y_pred.shape[0], y_pred.shape[1], -1)
#         y_true = y_true.contiguous().view(y_true.shape[0], y_true.shape[1], -1)

#         # Calculate IoU (Intersection over Union)
#         intersection = (y_pred * y_true).sum(dim=2)
#         union = (y_pred + y_true - y_pred * y_true).sum(dim=2)

#         iou = (intersection + self.smooth) / (union + self.smooth)

#         # The loss is 1 - IoU
#         return 1 - iou.mean()


def create_csv(image_dir: str) -> pd.DataFrame:
    """
    Create a CSV file with image paths and labels from the given directory.

    Args:
        image_dir (str): Directory containing images organized in subdirectories by label.

    Returns:
        pd.DataFrame: DataFrame containing image paths, mask paths and labels.
    """
    rows = []
    valid_exts = (".jpg", ".jpeg", ".png")

    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(valid_exts) and "mask" not in file.lower():
                image_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1]
                mask_file = file.replace(ext, f"_mask{ext}")
                mask_path = os.path.join(root, mask_file)

                # Infer label from immediate parent folder name
                label = os.path.basename(root)

                rows.append([image_path, mask_path, label])

    df = pd.DataFrame(rows, columns=["image_path", "mask_path", "label"])

    return df


def preprocess_image(img_path: str, sigma: list[float], weight: float = 0.04, denoise: bool = True) -> np.ndarray:
    """
    Preprocess an image by applying Gaussian gradient magnitude and denoising.

    Args:
        img_path (str): Path to the image file.
        weight (float): Weight for the Gaussian gradient magnitude.
        sigma (list[float]): Sigma values for Gaussian smoothing.
        denoise (bool): Whether to apply total variation denoising.

    Returns:
        np.ndarray: Preprocessed image as a NumPy array (Channel, Height, Width).
    """
    # Set default sigma if not provided
    if sigma is None:
        sigma = SIGMA

    # Read the image as grayscale and convert to float
    img = io.imread(img_path, as_gray=True)
    img = img_as_float(img)

    # Apply denoising if specified
    if denoise:
        img = denoise_tv_chambolle(img, weight=weight)

    # Apply Gaussian gradient magnitude for each sigma
    gradients = [gaussian_gradient_magnitude(img, sigma=s) for s in sigma]

    # Normalize each channel
    gradients = [(g - g.min()) / (g.max() - g.min()) for g in gradients]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Stack the gradients and the original image
    preprocessed_img = np.stack([img] + gradients, axis=0)

    return preprocessed_img


def _compute_seg_loss(seg_logits, region_logits, type_labels, masks, seg_loss_fn,
                      ce_loss_fn, ce_bg_seg_fn, alpha, device) -> torch.Tensor:
    """Computes the total segmentation loss for a batch.

    This function calculates a combined loss for segmentation, handling 'normal'
    and 'non-normal' (tumorous) samples differently.

    For non-normal samples, it computes a weighted average of a primary
    segmentation loss (e.g., Dice) and an auxiliary region-based cross-entropy loss.

    For normal samples, it applies a small background constraint loss to penalize
    any foreground predictions, encouraging the model to predict an empty mask.

    Args:
        seg_logits (torch.Tensor): The raw output logits from the segmentation head.
        region_logits (torch.Tensor): The raw output logits from an auxiliary region head.
        type_labels (torch.Tensor): Ground truth labels indicating the sample type
            (e.g., 0 for tumor type A, 1 for tumor type B, 2 for normal).
        masks (torch.Tensor): The ground truth segmentation masks.
        seg_loss_fn: The primary loss function for segmentation (e.g., DiceLoss).
        ce_loss_fn: The cross-entropy loss function for the region logits.
        ce_bg_seg_fn: The cross-entropy loss used for the background constraint on normal samples.
        alpha (float): The weighting factor to balance the main segmentation loss
            and the auxiliary region loss for non-normal samples.
        device (torch.device): The computation device (e.g., 'cuda' or 'cpu').

    Returns:
        torch.Tensor: A scalar tensor representing the final combined segmentation loss.
    """
    # Split batch by label
    non_normal = (type_labels == 0) | (type_labels == 1)
    normal = ~non_normal

    # SEGMENTATION
    seg_loss = torch.tensor(0.0, device=device)

    # 1) Usual segmentation loss
    if non_normal.any():
        seg_logits_non_normal = seg_logits[non_normal]
        region_logits_non_normal = region_logits[non_normal]
        masks_non_normal = masks[non_normal]
        target_indices_non_normal = masks_non_normal.squeeze(1).long()

        main_seg_loss = seg_loss_fn(seg_logits_non_normal, masks_non_normal)
        region_loss = ce_loss_fn(region_logits_non_normal, target_indices_non_normal)
        seg_loss = (1 - alpha) * main_seg_loss + alpha * region_loss

    # 2) Tiny background-only constraint on normal samples
    if normal.any() and LAMBDA_BG > 0.0:
        seg_logits_normal = seg_logits[normal]
        bg_target = torch.zeros_like(masks[normal].squeeze(1), dtype=torch.long)
        bg_ce = ce_bg_seg_fn(seg_logits_normal, bg_target)
        seg_loss += LAMBDA_BG * bg_ce

    return seg_loss


def _compute_cls_loss(presence_loss_fn, type_loss_fn, presence_logits,
                      presence_labels, type_logits, type_labels, device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes classification losses for tumor presence and tumor type.

    The presence loss is computed for all samples. The type loss is only computed for samples where tumors are present
    (presence_labels == 1). If no tumors are in the batch, type_loss is set to 0.0.

    Args:
        presence_loss_fn: Loss function for tumor presence classification.
        type_loss_fn: Loss function for tumor type classification.
        presence_logits: Logits for tumor presence.
        presence_labels: Ground truth labels for tumor presence (0 or 1).
        type_logits: Logits for the tumor type.
        type_labels: Ground truth labels for the tumor type.
        device: The device (e.g., 'cuda' or 'cpu') to create tensors on.

    Returns:
        A tuple containing the presence loss and type loss, both as torch.Tensor.
    """
    # 1) Tumor presence
    presence_loss = presence_loss_fn(presence_logits, presence_labels.unsqueeze(1))

    # 2) Tumor type
    tumor_mask = presence_labels == 1
    if tumor_mask.any():  # Only compute if there are tumors in the batch
        type_loss = type_loss_fn(type_logits[tumor_mask], type_labels[tumor_mask].unsqueeze(1))
    else:
        type_loss = torch.tensor(0.0, device=device)

    return presence_loss, type_loss


def process_segmentation(seg_logits, presence_preds, target_indices, dice_fn, hd95_fn, iou_fn):
    """Processes segmentation predictions to compute Dice and HD95 metrics.

    This function converts segmentation logits to prediction masks and then
    calculates the Dice score, IoU and 95th percentile Hausdorff Distance (HD95).

    It includes special logic to handle different scenarios:

    - If the classification model predicts a sample as 'NORMAL', the corresponding
      segmentation mask is forced to be empty (all background).
    - It correctly handles metric calculation for edge cases, such as when
      both the ground truth and prediction masks are empty (perfect score) or
      when one is empty and the other is not (horrible score).

    Args:
        seg_logits (torch.Tensor): The raw logits from the segmentation model.
        presence_preds (torch.Tensor): The predicted presence labels (0 for NORMAL, 1 for TUMOR).
        target_indices (torch.Tensor): The ground truth segmentation masks as integer labels.
        dice_fn: A callable function that computes the Dice score.
        hd95_fn: A callable function that computes the 95th percentile Hausdorff Distance.
        iou_fn: A callable function that computes the Intersection over Union (IoU).

    Returns:
        tuple[list[float], list[float], list[float]]: A tuple containing three lists:
        one for the Dice scores, one for the HD95 scores and one for IoU scores of the samples
        in the batch.
    """
    dice_scores, hd95_scores, iou_scores = [], [], []

    # Segmentation predictions
    seg_preds = torch.argmax(seg_logits, dim=1, keepdim=True)

    # If NORMAL is predicted, force the mask to background (0)
    if (presence_preds == NORMAL).any():
        normal_mask = (presence_preds == NORMAL).view(-1, 1, 1, 1)
        seg_preds = torch.where(normal_mask, torch.zeros_like(seg_preds), seg_preds)

    # One hot encoding for segmentation metrics
    seg_preds_one_hot = one_hot(seg_preds.squeeze(1).long(), num_classes=2).permute(0, 3, 1, 2).float()
    masks_one_hot = one_hot(target_indices, num_classes=2).permute(0, 3, 1, 2).float()

    # Foreground presence checks
    ground_truth_present = masks_one_hot[:, 1].sum(dim=(1, 2)) > 0
    predicted_present = seg_preds_one_hot[:, 1].sum(dim=(1, 2)) > 0

    # If both are empty: DICE = 1, HD95 = 0
    both_empty = ~ground_truth_present & ~predicted_present
    if both_empty.any():
        n = int(both_empty.sum().item())
        dice_scores.extend([1.0] * n)
        hd95_scores.extend([0.0] * n)
        iou_scores.extend([1.0] * n)

    # If ground truth is empty and the prediction is not: DICE = 0, skip HD95
    gt_empty_pred_nonempty = ~ground_truth_present & predicted_present
    if gt_empty_pred_nonempty.any():
        n = int(gt_empty_pred_nonempty.sum().item())
        dice_scores.extend([0.0] * n)
        iou_scores.extend([0.0] * n)

    # Ground truth is not empty
    valid = ground_truth_present
    if valid.any():
        seg_pred_valid = seg_preds_one_hot[valid]
        seg_gt_valid = masks_one_hot[valid]

        # Calculate metrics
        dice = dice_fn(seg_pred_valid, seg_gt_valid).mean().item()
        hd95 = hd95_fn(seg_pred_valid, seg_gt_valid).mean().item()
        iou = iou_fn(seg_pred_valid, seg_gt_valid).mean().item()

        if np.isfinite(dice):
            dice_scores.append(dice)
        if np.isfinite(hd95):
            hd95_scores.append(hd95)
        if np.isfinite(iou):
            iou_scores.append(iou)

    return dice_scores, hd95_scores, iou_scores


def train(model: Module, optimizer: Optimizer, seg_loss_fn, presence_loss_fn, type_loss_fn, train_loader: DataLoader,
          val_loader: DataLoader, device: torch.device = torch.device("cpu"), epochs: int = 50, patience: int = 5,
          save_model: bool = True, save_path: str = "BestModel.pt", lr_decay_factor: float = 0.1, lr_patience: int = 2,
          min_lr: float = 1e-6, alpha: float = 0.4) -> dict:
    """
    Trains a model for multi-task learning.
    
    This function combines segmentation, tumor presence and tumor type classification tasks.
    Uses both training and validation datasets and includes mechanisms such as early stopping,
    learning rate scheduling and model checkpointing.

    :param model: The model to be trained, expected to have segmentation, tumor presence
        and tumor type classification outputs.
    :type model: torch.nn.Module
    :param optimizer: The optimizer used for updating the model parameters.
    :type optimizer: torch.optim.Optimizer
    :param seg_loss_fn: The loss function for segmentation predictions.
    :type seg_loss_fn: callable
    :param presence_loss_fn: The loss function for tumor presence classification.
    :type presence_loss_fn: callable
    :param type_loss_fn: The loss function for tumor type classification.
    :type type_loss_fn: callable
    :param train_loader: DataLoader for the training dataset.
    :type train_loader: torch.utils.data.DataLoader
    :param val_loader: DataLoader for the validation dataset.
    :type val_loader: torch.utils.data.DataLoader
    :param device: The device to train the model on. Defaults to CPU.
    :type device: torch.device
    :param epochs: The maximum number of epochs for training. Defaults to 20.
    :type epochs: int
    :param patience: Number of epochs with no validation improvement before early stopping.
    :type patience: int
    :param save_model: Flag to determine whether to save the best model during training.
    :type save_model: bool
    :param save_path: Path to save the best model if save_model is True. Defaults to "BestModel.pt".
    :type save_path: str
    :param lr_decay_factor: The factor by which the learning rate is reduced when a plateau in
        validation loss is detected. Defaults to 0.1.
    :type lr_decay_factor: float
    :param lr_patience: Number of epochs with no validation improvement before reducing
        the learning rate. Defaults to 2.
    :type lr_patience: int
    :param min_lr: The minimum learning rate after reductions. Defaults to 1e-6.
    :type min_lr: float
    :param alpha: Weight to combine segmentation loss components. Defaults to 0.4.
    :type alpha: float
    :return: A dictionary containing training and validation metrics including losses,
        classification and segmentation metrics.
    :rtype: dict
    """
    model.to(device)

    train_losses, val_losses = [], []
    seg_dice_scores, seg_hd95_scores, seg_iou_scores = [], [], []
    presence_auc_scores, type_auc_scores = [], []

    best_performance = float("-inf")
    early_stop_counter = 0

    # Set up a learning rate scheduler that reduces LR when validation loss plateaus
    scheduler = ReduceLROnPlateau(optimizer, factor=lr_decay_factor, patience=lr_patience, min_lr=min_lr)

    # Dice metric measures boundary similarity (higher is better)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    # IoU metric measures boundary similarity (higher is better)
    iou_metric = MeanIoU(include_background=False, reduction="mean")
    # Hausdorff Distance (95th percentile) measures boundary similarity (lower is better)
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    # Region loss functions
    ce_loss = CrossEntropyLoss()
    ce_bg_seg = CrossEntropyLoss()

    for epoch in range(epochs):
        training_loss = 0.0
        model.train()  # Set model to training mode - enables dropout, batch normalization updates, etc.

        # Training Loop
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for inputs, (masks, presence_labels, type_labels) in train_loader:
                inputs, masks = inputs.to(device), masks.to(device)
                presence_labels, type_labels = presence_labels.to(device), type_labels.to(device)

                optimizer.zero_grad()  # Clear gradients from the previous batch to prevent accumulation
                seg_logits, region_logits, presence_logits, type_logits = model(inputs)  # Forward pass

                # SEGMENTATION LOSS
                seg_loss = _compute_seg_loss(seg_logits, region_logits, type_labels, masks,
                                             seg_loss_fn, ce_loss, ce_bg_seg, alpha, device)

                # CLASSIFICATION LOSS
                presence_loss, type_loss = _compute_cls_loss(presence_loss_fn, type_loss_fn, presence_logits,
                                                             presence_labels, type_logits, type_labels, device)
                cls_loss = presence_loss + 1.2 * type_loss

                # Combine losses with weighting - decides the importance of each task
                loss = SEG_WEIGHT * seg_loss + CLS_WEIGHT * cls_loss

                loss.backward()  # Backward pass - compute gradients
                optimizer.step()  # Update model parameters based on gradients
                training_loss += loss.item()  # Accumulate batch loss for epoch average

                pbar.update(1)  # Update progress bar

        # Calculate average training loss for the epoch
        training_loss /= len(train_loader)
        train_losses.append(training_loss)

        # Validation Loop
        val_loss = 0.0
        # For classification metrics calculation
        label_presence, presence_probs = [], []
        label_type, type_probs = [], []
        dice_scores, hd95_scores, iou_scores = [], [], []  # For segmentation quality assessment

        model.eval()  # Set model to evaluation mode - disables dropout, uses running stats for batch norm
        with torch.no_grad():  # Disable gradient calculation for validation - saves memory and speeds up inference
            for inputs, (masks, presence_labels, type_labels) in val_loader:
                inputs, masks = inputs.to(device), masks.to(device)
                presence_labels, type_labels = presence_labels.to(device), type_labels.to(device)

                seg_logits, region_logits, presence_logits, type_logits = model(inputs)  # Forward pass

                target_indices = masks.squeeze(1).long()  # Prepare target indices for CE Loss

                # Calculate segmentation validation loss
                main_seg_loss = seg_loss_fn(seg_logits, masks)
                region_loss = ce_loss(region_logits, target_indices)
                seg_loss = (1 - alpha) * main_seg_loss + alpha * region_loss

                # Calculate classification validation loss
                presence_loss, type_loss = _compute_cls_loss(presence_loss_fn, type_loss_fn, presence_logits,
                                                             presence_labels, type_logits, type_labels, device)
                cls_loss = presence_loss + 1.2 * type_loss

                loss = SEG_WEIGHT * seg_loss + CLS_WEIGHT * cls_loss
                val_loss += loss.item()

                # Classification predictions
                presence_proba = torch.sigmoid(presence_logits)
                type_proba = torch.sigmoid(type_logits)
                presence_preds = (presence_proba >= 0.5).long()

                # Segmentation metrics
                d, h, iou = process_segmentation(
                    seg_logits, presence_preds, target_indices, dice_metric, hd95_metric, iou_metric
                )
                dice_scores.extend(d)
                hd95_scores.extend(h)
                iou_scores.extend(iou)

                # Convert classification stats
                presence_probs.extend(presence_proba.squeeze(1).cpu().numpy())
                label_presence.extend(presence_labels.cpu().numpy())

                tumor_mask = presence_labels == 1
                type_probs.extend(type_proba[tumor_mask].squeeze(1).cpu().numpy())
                label_type.extend(type_labels[tumor_mask].cpu().numpy())

        # Calculate average validation loss
        val_loss /= max(1, len(val_loader))
        val_losses.append(val_loss)

        mean_dice = np.mean(dice_scores) if len(dice_scores) > 0 else np.nan
        mean_hd95 = np.mean(hd95_scores) if len(hd95_scores) > 0 else np.nan
        mean_iou = np.mean(iou_scores) if len(iou_scores) > 0 else np.nan

        label_presence, label_type = np.array(label_presence), np.array(label_type)
        presence_probs, type_probs = np.array(presence_probs), np.array(type_probs)

        # PR-AUC metric measures the tradeoff between sensitivity and specificity
        try:
            presence_auc = average_precision_score(label_presence, presence_probs, average="weighted")
            type_auc = average_precision_score(label_type, type_probs, average="weighted")
            pr_auc = presence_auc + 1.2 * type_auc
        except ValueError:
            # Handle case where some classes might not be present in the batch
            pr_auc, presence_auc, type_auc = np.nan, np.nan, np.nan

        # Calculate ROC curve for malignancy detection (binary classification)
        try:
            fpr, tpr, _ = roc_curve(label_type, type_probs)
        except ValueError:
            fpr, tpr, _ = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0])

        # Calculate specificity at 90% sensitivity - clinically relevant operating point
        # High sensitivity is critical for cancer detection to minimize false negatives
        spec_at_90sens = float("nan")
        if np.any(tpr >= 0.9):
            idx = np.where(tpr >= 0.9)[0][0]  # Find the first threshold with sensitivity ≥ 90%
            spec_at_90sens = 1.0 - fpr[idx]  # Convert FPR to specificity

        # Calculate sensitivity at 90% specificity - alternative operating point
        # High specificity reduces unnecessary biopsies/interventions
        sens_at_90spec = float("nan")
        spec = 1.0 - fpr  # Convert FPR to specificity
        if np.any(spec >= 0.9):
            idx = np.where(spec >= 0.9)[0][-1]  # Find the first threshold with specificity ≥ 90%
            sens_at_90spec = tpr[idx]

        seg_dice_scores.append(mean_dice)
        seg_hd95_scores.append(mean_hd95)
        seg_iou_scores.append(mean_iou)
        presence_auc_scores.append(presence_auc)
        type_auc_scores.append(type_auc)

        print(
            f"Train Loss: {training_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Dice: {mean_dice:.4f} | Val IoU: {mean_iou:.4f} | "
            f"Val HD95: {mean_hd95:.4f} | Presence AUC: {presence_auc:.4f} | "
            f"Type AUC: {type_auc:.4f} | Spec@90sens: {spec_at_90sens:.4f} | "
            f"Sens@90spec: {sens_at_90spec:.4f}"
        )

        scheduler.step(val_loss)  # Update learning rate based on validation loss

        # This combined metric ensures we select a model that performs well on both tasks
        performance = SEG_WEIGHT * (mean_dice if not np.isnan(mean_dice) else 0.0) + CLS_WEIGHT * (
            pr_auc if not np.isnan(pr_auc) else 0.0)
        if performance > best_performance:
            best_performance = performance
            early_stop_counter = 0  # Reset early stopping counter on improvement

            if save_model:
                # Save only the model parameters (state_dict) rather than the entire model
                torch.save(model.state_dict(), save_path)
                print(f"Model saved to {save_path}")
        else:
            early_stop_counter += 1

        # Early stopping mechanism to prevent overfitting
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

        # Clean up to free memory
        gc.collect()
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()

    return {
        "Train_Loss": train_losses,
        "Val_Loss": val_losses,
        "Seg_Dice_Scores": seg_dice_scores,
        "Seg_HD95_Scores": seg_hd95_scores,
        "Seg_IoU_Scores": seg_iou_scores,
        "Presence_AUC_Scores": presence_auc_scores,
        "Type_AUC_Scores": type_auc_scores,
    }


if __name__ == "__main__":
    print("Nothing to see here, lol")
