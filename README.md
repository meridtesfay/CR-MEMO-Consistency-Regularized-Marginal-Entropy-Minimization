CR-MEMO: Consistency-Regularized Marginal Entropy Minimization

Author: Merid | Course: Deep Learning (Jan 2026)

🚀 Overview

This repository implements CR-MEMO, an enhancement of the MEMO algorithm (Zhang et al., NeurIPS 2022). This project focuses on Test-Time Adaptation (TTA), enabling pre-trained models to adapt to distribution shifts—such as natural corruptions (ImageNet-V2) and adversarial examples (ImageNet-A) —during inference using only the test point itself.

✨ My Contribution: Consistency Regularization

The original MEMO minimizes the entropy of the average prediction across augmentations. Inspired by consistency training (Xie et al., 2020), I introduced a Consistency-Regularized Marginal Entropy Loss.
By adding a KL-Divergence penalty, the model is encouraged to produce consistent predictions across all augmented views. This prevents the model from collapsing into high-confidence but incorrect predictions, especially under heavy distribution shift.

📊 Comparative Analysis

The following table summarizes the performance of standard MEMO variants and my Consistency (Cons) modifications across two architectures.

Method,ImageNet-A (%),Δ,ImageNet-V2 (%),Δ
ResNet-50,,,,
    + Baseline,14.31,—,69.93,—
    + RRC (MEMO),18.44,"<font color=""green"">(+4.13)</font>",75.90,"<font color=""green"">(+5.97)</font>"
    + RRC_BN,24.25,"<font color=""green"">(+9.94)</font>",78.61,"<font color=""green"">(+8.68)</font>"
    + RRC_FLIP_BN,24.11,"<font color=""green"">(+9.80)</font>",78.72,"<font color=""green"">(+8.79)</font>"
    + Cons (My Mod),18.12,"<font color=""green"">(+3.81)</font>",76.02,"<font color=""green"">(+6.09)</font>"
    + Cons_BN,23.64,"<font color=""green"">(+9.33)</font>",78.66,"<font color=""green"">(+8.73)</font>"
    + Cons_BN_FLIP,24.11,"<font color=""green"">(+9.80)</font>",79.05,"<font color=""green"">(+9.12)</font>"
---,---,---,---,---
DenseNet-121,,,,
    + Baseline,2.15,—,61.99,—
    + RRC (MEMO),11.68,"<font color=""green"">(+9.53)</font>",73.92,"<font color=""green"">(+11.93)</font>"
    + RRC_BN,11.00,"<font color=""green"">(+8.85)</font>",73.66,"<font color=""green"">(+11.67)</font>"
    + RRC_FLIP_BN,11.07,"<font color=""green"">(+8.92)</font>",73.59,"<font color=""green"">(+11.60)</font>"
    + Cons (My Mod),5.68,"<font color=""green"">(+3.53)</font>",71.18,"<font color=""green"">(+9.19)</font>"
    + Cons_BN,11.03,"<font color=""green"">(+8.88)</font>",73.36,"<font color=""green"">(+11.37)</font>"
    + Cons_BN_FLIP,10.67,"<font color=""green"">(+8.52)</font>",73.32,"<font color=""green"">(+11.33)</font>"
