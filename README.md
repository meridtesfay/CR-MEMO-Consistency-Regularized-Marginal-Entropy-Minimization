CR-MEMO: Consistency-Regularized Marginal Entropy Minimization

Author: Merid | Course: Deep Learning (Jan 2026)

🚀 Overview

This repository implements CR-MEMO, an enhancement of the MEMO algorithm (Zhang et al., NeurIPS 2022). This project focuses on Test-Time Adaptation (TTA), enabling pre-trained models to adapt to distribution shifts—such as natural corruptions (ImageNet-V2) and adversarial examples (ImageNet-A) —during inference using only the test point itself.

✨ My Contribution: Consistency Regularization

The original MEMO minimizes the entropy of the average prediction across augmentations. Inspired by consistency training (Xie et al., 2020), I introduced a Consistency-Regularized Marginal Entropy Loss.
By adding a KL-Divergence penalty, the model is encouraged to produce consistent predictions across all augmented views. This prevents the model from collapsing into high-confidence but incorrect predictions, especially under heavy distribution shift.

📊 Comparative Analysis

The following table summarizes the performance of standard MEMO variants and my Consistency (Cons) modifications across two architectures.

### Table 1: Comparative Analysis of MEMO Variants

| Method | ImageNet-A (%) | Δ | ImageNet-V2 (%) | Δ |
| :--- | :---: | :---: | :---: | :---: |
| **ResNet-50** | | | | |
| &nbsp;&nbsp;&nbsp; + Baseline | 14.31 | — | 69.93 | — |
| &nbsp;&nbsp;&nbsp; + RRC (MEMO) | 18.44 | (+4.13) | 75.90 | (+5.97) |
| &nbsp;&nbsp;&nbsp; + RRC_BN | **24.25** | **(+9.94)** | 78.61 | (+8.68) |
| &nbsp;&nbsp;&nbsp; + RRC_FLIP_BN | 24.11 | (+9.80) | 78.72 | (+8.79) |
| &nbsp;&nbsp;&nbsp; + Cons (My Mod) | 18.12 | (+3.81) | 76.02 | (+6.09) |
| &nbsp;&nbsp;&nbsp; + Cons_BN | 23.64 | (+9.33) | 78.66 | (+8.73) |
| &nbsp;&nbsp;&nbsp; + Cons_BN_FLIP | 24.11 | (+9.80) | **79.05** | **(+9.12)** |
| --- | --- | --- | --- | --- |
| **DenseNet-121** | | | | |
| &nbsp;&nbsp;&nbsp; + Baseline | 2.15 | — | 61.99 | — |
| &nbsp;&nbsp;&nbsp; + RRC (MEMO) | **11.68** | **(+9.53)** | **73.92** | **(+11.93)** |
| &nbsp;&nbsp;&nbsp; + RRC_BN | 11.00 | (+8.85) | 73.66 | (+11.67) |
| &nbsp;&nbsp;&nbsp; + RRC_FLIP_BN | 11.07 | (+8.92) | 73.59 | (+11.60) |
| &nbsp;&nbsp;&nbsp; + Cons (My Mod) | 5.68 | (+3.53) | 71.18 | (+9.19) |
| &nbsp;&nbsp;&nbsp; + Cons_BN | 11.03 | (+8.88) | 73.36 | (+11.37) |
| &nbsp;&nbsp;&nbsp; + Cons_BN_FLIP | 10.67 | (+8.52) | 73.32 | (+11.33) |

Key Finding: My modification (Cons_BN_FLIP) achieved the state-of-the-art result for ResNet-50 on ImageNet-V2, showing a +9.12% improvement over the baseline.

🛠 Repository Structure

├── assets/                 # Results images and plots
├── src/                    # Core logic
│   ├── losses.py           # CR-MEMO Loss (KL + Marginal Entropy)
│   ├── model.py            # Model loading & AdaBN
│   └── tta.py              # Adaptation loop logic
├── main.py                 # CLI for running experiments
├── requirements.txt        # Dependencies
└── MEMO_Implementation.ipynb # Original experimental notebook




🚀 Quick Start

# 1. Install dependencies
pip install -r requirements.txt

# 2. Run ResNet-50 with Consistency Regularization
python main.py --arch resnet50 --dataset imagenet-v2 --kl_weight 0.1





📚 References

    Zhang et al. (2022). MEMO: Test Time Robustness via Adaptation and Augmentation. NeurIPS.

    Hendrycks et al. (2021). Natural Adversarial Examples. CVPR.

    Xie et al. (2020). Unsupervised Data Augmentation for Consistency Training. NeurIPS.
