## NYCU Selected Topics in Visual Recognition using Deep Learning 2025 Spring HW4


### Student Information

- **Student ID:** 413540004
- **Student Name:** Phan Nguyen Minh Thao (潘阮明草)

### Introduction
Blind image restoration aims to recover high-quality images from degraded ones without knowing the degradation type. PromptIR introduces a novel prompt-based Transformer that learns adaptive restoration patterns for multiple degradations simultaneously. It achieves strong generalization and high-quality output using a multi-stage Transformer encoder-decoder structure.

#### Installation & Dependencies
Ensure you have Python 3.9+ installed. Install the required dependencies:

- Python: 3.9.21
- PyTorch: 2.5.1+cu124
- NumPy: 2.0.2
- Lightning: 2.0.1
- Pillow:9.4.0
- Piq: 0.8.0
- Einops: 0.6.0
- Scikit-image: 0.19.3

### Model Details

The PromptIR model is composed of the following components:
- Overlap Patch Embedding: Extracts low-level features from input images.
- Hierarchical Transformer Encoder: Multi-scale attention for feature extraction.
- PromptGen Modules: Dynamically generate degradation-aware prompts to condition the decoder.
- Hierarchical Decoder: Reconstructs clean images from latent features.
- Refinement Module: Applies final enhancements for image quality.

### Data Preparation

#### Input Format

Training and testing data should be organized as follows:

```bash
hw4-data/
├── train/
│   ├── degraded/
│   │   ├── rain-1.png
│   │   ├── ...
│   │   ├── rain-1600.png
│   │   ├── snow-1.png
│   │   ├── ...
│   │   ├── snow-1600.png
│   ├── clean/
│   │   ├── rain_clean-1.png
│   │   ├── ...
│   │   ├── rain_clean-1600.png
│   │   ├── snow_clean-1.png
│   │   ├── ...
│   │   ├── snow_clean-1600.png
├── test/
│   └── degraded/
│       ├── 0.png
│       ├── ...
│       ├── 99.png
```
The test images are named generically (e.g., 0.png to 99.png) and do not reveal the type of degradation.

**Note**:

- Training set uses ```bash degraded/``` and corresponding ```bash clean/``` images.
- Filenames must follow the pattern:
```bash rain-xxx.png``` → ```bash rain_clean-xxx.png```
```bash snow-xxx.png``` → ```bash snow_clean-xxx.png```

### Training

Run the following command to train on derain and desnow tasks:

```bash
python 413540004.py \
  --epochs 200 \
  --batch_size 4 \
  --lr 2e-4 \
  --de_type derain desnow \
  --patch_size 128 \
  --num_workers 4 \
  --ckpt_dir train_ckpt
```

- Checkpoints will be saved in: ```bash train_ckpt/```
- TensorBoard logs will be saved in: ```bash logs/```

### Evaluation

The model logs the following metrics on the validation set:

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM Loss
- Total Loss (L1 + SSIM)

These are recorded automatically using ```bash LightningModule.log```.

### Submission

To generate predictions on the test set and prepare the submission:

```bash
python 413540004_submission.py \
  --test_path hw4-data/test/degraded/ \
  --output_path output \
  --ckpt_dir train_ckpt \
  --ckpt_name model.ckpt
```

Outputs:

- Restored images saved in ```bash output/```

- ```bash pred.npz``` containing output tensors

- A timestamped ZIP archive: ```bash output/YYYYMMDD__HHMMSS.zip```

### Performance snapshot
![alt text](snapshot.png)

### Acknowledgements

This project is based on the [PromptIR](https://github.com/va1shn9v/PromptIR)  architecture from the original paper. We thank the authors for their insightful work.

###  Reference

Potlapalli, V., Zamir, S. W., Khan, S. H., & Shahbaz Khan, F. (2023). Promptir: Prompting for all-in-one image restoration. Advances in Neural Information Processing Systems, 36, 71275-71293.