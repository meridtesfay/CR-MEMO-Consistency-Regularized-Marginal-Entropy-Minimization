CR-MEMO: Consistency-Regularized Marginal Entropy Minimization

Author: Merid | Course: Deep Learning (Jan 2026)

🚀 Overview

This repository implements CR-MEMO, an enhancement of the MEMO algorithm (Zhang et al., NeurIPS 2022). This project focuses on Test-Time Adaptation (TTA), enabling pre-trained models to adapt to distribution shifts—such as natural corruptions (ImageNet-V2) and adversarial examples (ImageNet-A) —during inference using only the test point itself.

✨ My Contribution: Consistency Regularization

The original MEMO minimizes the entropy of the average prediction across augmentations. Inspired by consistency training (Xie et al., 2020), I introduced a Consistency-Regularized Marginal Entropy Loss.
By adding a KL-Divergence penalty, the model is encouraged to produce consistent predictions across all augmented views. This prevents the model from collapsing into high-confidence but incorrect predictions, especially under heavy distribution shift.
