# PRECISE Abreast Challenge — Multi-task OCA Model

This repository contains an implementation for the PRECISE Abreast Challenge: a multi-task model that performs breast tumor presence classification, tumor type classification (benign vs malignant), and tumor segmentation from 2D breast ultrasound images.

The implementation is in Python and PyTorch. Main artifacts:

- `main.ipynb` — notebook used for experiments (data loading, training, evaluation, inference, and submission generation).
- `src/` — source code including model definition, datasets, and utilities.
  - `src/modules.py` — backbone / model building blocks (ResUNet used as backbone).
  - `src/mtl_oca.py` — `MultiTaskOCAModel` combining segmentation + classification heads.
  - `src/utils.py` — dataset classes (`BreastDataset`, `BreastTestDataset`), losses (`IoULoss`), training loop (`train`), preprocessing helpers (`create_csv`, `process_segmentation`, `JointTransform`), and other utilities.
- `models/` — saved model checkpoints (example: `MultiTaskOCAModel.pt`).
- `submission/` — outputs: `classification.csv`, `segmentation/` masks, and confusion matrix.

## Quick start

1. Create a Python environment and install dependencies listed in `requirements.txt`.

2. Run the notebook to train or evaluate: open `main.ipynb` and run the cells.

3. To generate a submission, use the inference section of `main.ipynb` which writes `submission/classification.csv` and segmentation masks to `submission/segmentation/`.

## Project overview and approach

The goal is to produce a single model that simultaneously:

- Predicts whether a tumor is present in the image (binary classification).
- If a tumor is present, classifies it as benign or malignant (binary classification among tumor images).
- Produces a segmentation mask for tumors (pixel-wise segmentation).

To achieve this we use a multi-task architecture built on a ResUNet backbone with separate heads for segmentation and classification.

### Data and preprocessing

- Dataset structure: images are stored under `images/Train/<label>/` and `images/Val/<label>/` where `<label>` is one of `benign`, `malignant`, or `normal`.

- `create_csv` scans a root folder and builds a DataFrame with columns: `image_path`, `mask_path`, `label`. `label` is mapped to integer classes via: `benign -> 0`, `malignant -> 1`, `normal -> 2`.

- Data cleaning: `verify_and_clean` (in the notebook) checks file existence and drops rows with missing image/mask files.

- Transformations and augmentation: A `JointTransform` class in `src.utils` applies coordinated transformations to the image and mask such as resize (to `IMAGE_SIZE`), random flips, rotations and intensity normalization. Augmentations are applied to training data only; validation/test data use deterministic resizing and normalization.

- Labels used at training:
  - Presence label: derived from `label` where `benign` and `malignant` map to tumor present (1), and `normal` maps to not present (0).
  - Type label: among images with tumor present, `benign` -> 0, `malignant` -> 1.

- Image normalization: images are converted to tensors and normalized to [0, 1] or standard z-score depending on transformations in `JointTransform`.

### Model architecture

- Backbone: ResUNet implemented in `src/modules.py`. It follows a U-Net style encoder-decoder with skip connections and channel sizes controlled by the configuration `FILTERS` in the notebook.

- Multi-task head: `MultiTaskOCAModel` in `src/mtl_oca.py` attaches three heads on top of the shared backbone:
  - Segmentation head: outputs per-pixel logits for background and tumor classes. During training the segmentation output is compared against masks (one-hot/softmax style with `to_onehot_y=True` where necessary).
  - Presence head: a binary classifier predicting tumor presence (single logit output per image). Uses Focal Loss during training for class imbalance handling.
  - Type head: a binary classifier (benign vs malignant) that is only meaningful for images where a tumor is present. During training the type head loss is computed only on tumor-containing images. Also trained with Focal Loss.

- OCA (Object-Contextual Attention) components: the model includes an attention modules to better incorporate context for segmentation and classification tasks. This improves localization and type discrimination when tumors are small or noisy.

### Losses and training strategy

- Segmentation loss: `DiceLoss` from MONAI is used for segmentation supervision. The notebook uses `DiceLoss(include_background=False, to_onehot_y=True, softmax=True)` to focus on tumor regions and handle multi-class logits.

- Classification losses: `FocalLoss` from MONAI is used for both presence and type classification. Different alpha balances are used to account for label imbalance:
  - Presence loss: FocalLoss(gamma=2.0, alpha=0.07)
  - Type loss: FocalLoss(gamma=2.0, alpha=0.70)

- Combined loss: total loss is sum or weighted sum of segmentation loss + presence loss + type loss. Weights can be tuned to prioritize segmentation versus classification.

- Optimizer and scheduler: AdamW is used as the optimizer (lr=1e-3 in the notebook). A learning rate decay is applied on the best validation metric plateau with `lr_decay_factor=0.1` after patience epochs.

- Early stopping: training includes a patience mechanism (default patience=5) that stops training when the validation segmentation metric doesn't improve.

- Training settings used in experiments:
  - Batch size: 32
  - Image size: 128x128
  - Epochs: 100 (early stopping typically triggers earlier)
  - Seed: 42 (deterministic runs via manual seeding and torch.backends.cudnn.deterministic)

### Evaluation metrics

- Segmentation: Dice Score (mean, exclude background), 95th-percentile Hausdorff Distance (HD95), Mean IoU.
- Classification: Precision-Recall Area Under Curve (PR-AUC), Sensitivity@90%Specificity, Specificity@90%Sensitivity and confusion matrices.

Evaluation is performed per-fold inside a Stratified K-Fold (default 5 folds) experiment. Final reported metrics are averages across folds.

### Inference and submission

- The notebook loads a saved checkpoint `models/MultiTaskOCAModel.pt` and runs inference on validation or test images. It applies sigmoid to classification logits and selects thresholds by maximizing F1 on validation predictions.

- For the type prediction, if the presence head predicts `normal` for an image, the final class is forced to `normal` and the segmentation mask is set to background.

- Segmentation outputs are resized back to original image size and saved as binary masks (0 -> 0, 1 -> 255) into `submission/segmentation/`.

- A final `submission/classification.csv` is written with columns: `image_id`, `label` (string label: `Benign`, `Malignant`, `Normal`).

## Techniques and design choices

- Multi-task learning: sharing a backbone between segmentation and classification helps the model learn better feature representations with limited data.

- Focal loss: handles class imbalance by down-weighting easy negatives and focusing training on hard examples.

- Dice loss for segmentation: optimizes directly for overlap-based metrics (Dice/IoU) used in evaluation.

- Stratified K-Fold cross validation: ensures class balance across folds for robust validation and threshold selection.

- Seeded deterministic data loading: reproducible experiments across runs.

- Object Contextual Attention (OCA) Module: provide improved context modeling which helps locating small lesions.

## Classification Results

- Precision: 0.7916
- Recall: 0.7934
- AUC: 0.7871
- F1 Score: 0.7921

![Confusion Matrix](submission\confusion_matrix.png)

## Segmentation Results

- Dice Score: 0.9159
- HD95: 0.8085
- IoU: 0.8938

![Benign Image](submission\PACE_00001_001_BUSBRA.png)
![Benign Segmentation](submission\segmentation\PACE_00001_001_BUSBRA_mask.png)

## How to reproduce

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Inspect and run `main.ipynb` in Jupyter or VS Code. Key cells:
   - Data preparation and `create_csv` usage.
   - `stratified_kfold_train(...)` to run training and CV.
   - Inference section that loads `models/MultiTaskOCAModel.pt` and produces `submission/`.

3. To train from scratch in a script, adapt the notebook into a `.py` script and call `train(...)` from `src.utils`.

## Dependencies

- See `requirements.txt` for exact versions. Key dependencies include: PyTorch, MONAI, torchvision, scikit-learn, numpy, pandas, Pillow, tqdm, matplotlib.
- Contact the author of this project for rights to dataset access.

## Files of interest

- `main.ipynb` — orchestration, CV, and inference pipeline.
- `src/utils.py` — dataset classes, transforms, training loop, and helper functions.
- `src/mtl_oca.py` — multi-task model glue code.
- `src/modules.py` — ResUNet, layers and attention blocks.
- `models/` — saved checkpoints.

## Future Improvements

- Hyperparameter sweep: tune segmentation/classification loss weights, learning rate, and focal loss alpha parameters.
- Larger input sizes (e.g., 256x256) can improve segmentation detail at the cost of memory and training time.
- Experimenting with different variations of the ResUNet backbone such as AttentionUNet, R2UNet, AttentionR2UNet, ResUNet++.
- Post-processing: small morphological operations on segmentation masks to remove spurious detections.
- Model explainability: use Grad-CAM or attention visualization to inspect what the model uses for decisions.
