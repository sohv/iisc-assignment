# IISc Assignment

This repository contains the code for the assignment given as part of ViSTA Lab application at IISc. It includes the implementations of the following tasks:
1. Vision Transformer (ViT) for image classification on the CIFAR-10 dataset.
2. Text-driven Image Segmentation via SAM2

## Setup

To set up the environment to run the code, follow these steps:

1. Install the required dependencies (pip install <package_name>).
   ```bash
   torch
   opencv-python
   pillow
   matplotlib
   numpy
   tqdm
   ```
2. Open the Jupyter notebook files `q1.ipynb` and `q2.ipynb` and run them in order.

3. For the input images in `q2.ipynb`, you can download the used images from [here](https://drive.google.com/drive/folders/1NOW0IEx8mx6COBwpM-YfDmjnGMrbEmSF?usp=sharing).

## Task 1: q1.ipynb

This notebook implements a Vision Transformer (ViT) model, built from scratch using PyTorch, for image classification on the CIFAR-10 dataset. The model is trained and evaluated, and the training and test accuracy are recorded.

### Configuration

The model configuration used achieves optimal results without risking overfitting, dimensionality issues or unstable training. The configuration is as follows:

```bash
# architecture
img_size = 32
patch_size = 4
num_patches = 64
embed_dim = 384
depth = 12
num_heads = 12
mlp_ratio = 4

# regularization
dropout = 0.1
attn_dropout = 0.1
stochastic_depth = 0.1      # drop path rate

# training
batch_size = 128
epochs = 300
learning_rate = 3e-4
weight_decay = 0.05
warmup_epochs = 10
label_smoothing = 0.1
grad_clip = 1.0

# Optimizer: AdamW
```

### Analysis and Design Choices

#### 1. **Patch Size selection**

The patch size is set to 4, resulting in an 8x8 grid of patches for the 32x32 input images. This choice balances the need for capturing local features while maintaining a manageable sequence length for the transformer. The rationale is that CIFAR-10 images are small, so we need fine-grained patches to capture sufficient detail. The patch sizes 2,4,8 and 16 were tested and 4 was found to be optimal.

| Patch Size | Num Patches | Trade-off |
|------------|-------------|-----------|
| 2×2 | 256 | Too many patches, high computational cost |
| **4×4** | **64** | **Optimal balance** |
| 8×8 | 16 | Too few patches, loses spatial detail |
| 16×16 | 4 | Severe information loss |

#### 2. **Depth vs Width Trade-offs**

Increasing the depth for the vision transformer improves the performance upto 12 layers. Adding more layers beyond 12 leads to diminishing returns and a higher risk of overfitting. While increasing the embedding dimension (width) helps, depth has a more significant impact on performance for this dataset [1]. The different configurations tested were:

| Config | Depth | Embed Dim | Params | 
|--------|-------|-----------|--------|
| Small | 6 | 256 | ~2M | 
| **Optimal** | **12** | **384** | **~4.7M** | 
| Large | 16 | 512 | ~12M | 
| XL | 20 | 768 | ~35M | 

Hence, the configuration of 12 layers with an embedding dimension of 384 hits a sweet spot between accuracy and model complexity (number of parameters).

#### 3. **Data Augmentation Strategy**

A combination of random cropping, horizontal flipping, AutoAugment and Random Erasing was used for data augmentation. This strategy helps improve the model's generalization by exposing it to a variety of transformations, which is particularly important given the relatively small size of the CIFAR-10 dataset.

```python
1. RandomCrop(32, padding=4)      
2. RandomHorizontalFlip(p=0.5)    
3. AutoAugment(CIFAR10)           
4. RandomErasing(p=0.25)       
```

| Augmentation | Benefit |
|--------------|---------|
| Baseline | None|
| + Crop/Flip | Basic generalization |
| + AutoAugment | Complex transformations |
| + RandomErasing | Robustness to occlusion |

**Key Insight**: AutoAugment provides the largest single improvement [2], as it uses learned augmentation policies optimized for CIFAR-10.

#### 4. **Regularization Techniques**

| Technique |  Purpose |
|-----------|---------|
| Dropout (0.1) | Prevent co-adaptation |
| Stochastic Depth (0.1) | Deep network training |
| Weight Decay (0.05) | L2 regularization |
| Label Smoothing (0.1) | Confidence calibration |

**Stochastic Depth** (Drop Path) [3] randomly drops entire residual connections during training, thereby improving gradient flow in deep networks.

### Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **92.20%** |
| **Train Accuracy** | **95.42%** |
| **Test Loss** | **0.7351** |
| **Train Loss** | **0.6031** |
| **Parameters** | **~4.7M** |

Furthermore, the training and test accuracy over epochs shows steady improvement and convergence without overfitting.

|Epochs|Train Accuracy (%)|Test Accuracy (%)|
|------|------------------|-----------------|
| 50   | 74.47            | 82.52           |
| 100  | 84.72            | 88.33           |
| 150  | 89.68            | 90.08           |
| 200  | 92.72            | 91.05           |
| 250  | 94.64            | 91.79           |
| 300  | 95.42            | 92.20           |

## Task 2: q2.ipynb

This notebook implements text-prompted segmentation using the Segment Anything Model 2 (SAM 2) combined with GroundingDINO for zero-shot object detection. The pipeline accepts natural language descriptions (e.g. 'person', 'dog', 'car') and generates precise segmentation masks.

The bonus analysis includes video object segmentation with temporal mask propagation across frames.

### Pipeline Overview

#### Image Segmentation pipeline

```bash
Text Prompt -> GroundingDINO -> Bounding Boxes -> SAM 2 -> Precise Masks
```

1. Text Input: User provides a natural language description (e.g., "person wearing red shirt")
2. Object detection: GroundingDINO detects objects matching the text description
3. Region proposal: The bounding boxes serve as prompts for SAM 2
4. Mask generation: SAM 2 produces high-quality segmentation masks
5. Visualization: Masks overlaid on original image with color coding

#### Video Segmentation pipeline

```bash
First Frame Detection -> SAM 2 Video Predictor -> Temporal Propagation -> Video Output
```

1. Initial detection: Detect the target object in first frame using text prompt
2. Mask initialization: Generate an initial mask with SAM 2
3. Temporal propagation: SAM 2's video predictor tracks the object across frames
4. Video reconstruction: Create output video with mask overlays

### Example Results

**Image Segmentation**

- Input: Image + "person"
- Output: Precise person segmentation with smooth boundaries
- Performance: ~2-3 seconds per image on GPU

**Video Segmentation**

- Input: 30-second video + "car"
- Output: Video with tracked car segmentation
- Performance: ~0.5-1 second per frame on GPU

### Limitations

- Text Ambiguity: Vague prompts can lead to incorrect detections
- Object Scale: Very small or large objects may be missed as GroundingDINO has detection threshold limitations.
- Occlusions: Heavily occluded objects may not be segmented accurately.
- Video Drift: Long videos may experience mask drift due to cumulative errors in propagation.

### Best Practices

1. Use Specific Prompts: 'golden retriever' > 'dog' > 'animal'
2. Adjust Thresholds: Set the threshold lower for more detections, higher for precision
3. Batch Processing: Process multiple frames together when possible

## References

[1] Wu, Gent. "Powerful Design of Small Vision Transformer on CIFAR10." arXiv preprint arXiv:2501.06220 (2025).

[2] Cubuk, Ekin D., et al. "Autoaugment: Learning augmentation policies from data." arXiv preprint arXiv:1805.09501 (2018).

[3] Huang, Gao, et al. "Deep networks with stochastic depth." European conference on computer vision. Cham: Springer International Publishing, 2016.

