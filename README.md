# Hybrid Transformer-ML-CNN-BiLSTM Model with WGAN-GP for Network Intrusion Detection

**A robust deep learning framework for detecting network intrusions using synthetic data augmentation and advanced feature extraction.**

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

## Project Overview

This project implements a hybrid deep learning model for Network Intrusion Detection Systems (NIDS), leveraging the UNSW-NB15 dataset. The framework addresses critical challenges in cybersecurity—such as class imbalance and poor generalization—by combining Wasserstein Generative Adversarial Networks with Gradient Penalty (WGAN-GP) for synthetic data augmentation, and a Transformer-ML-CNN-BiLSTM architecture for advanced feature extraction and classification.

---

## Key Features

- **WGAN-GP for Data Augmentation:** Generates realistic synthetic samples of underrepresented attack classes to address data imbalance.
- **Transformer Encoder:** Captures global dependencies and contextual relationships in network traffic.
- **Multi-Scale CNN:** Extracts hierarchical spatial features from raw traffic data.
- **Bidirectional LSTM (BiLSTM):** Models temporal dependencies in network sequences for enhanced anomaly detection.
- **Focal Loss:** Optimizes training by prioritizing hard-to-classify samples.
- **Cosine Annealing Learning Rate Scheduling:** Improves model convergence and generalization.
- **Dtype Handling:** Ensures numerical stability and compatibility by managing float32 and mixed precision settings.
- **Early Stopping and Reduced Learning Rate:** Prevents overfitting and accelerates convergence.
---

## Architecture

The proposed model integrates several advanced deep learning modules:

1. **WGAN-GP:** Synthetic data generation for minority class augmentation.
2. **Transformer Encoder:** Multi-head self-attention for global feature extraction.
3. **Multi-Scale CNN:** Parallel convolutional layers with different kernel sizes for local and global feature learning.
4. **Bidirectional LSTM:** Sequential modeling of network traffic for temporal dependencies.
5. **Focal Loss and Cosine Annealing:** Optimized training and improved classification performance.

---

## Dataset

- **UNSW-NB15:** A comprehensive network traffic dataset containing both normal and malicious traffic, widely used for benchmarking NIDS models.
- **Feature Selection:** Categorical features are encoded, and numerical features are normalized.
- **Data Augmentation:** Synthetic attack samples are generated using WGAN-GP to balance the dataset.

---

## References

**Related Papers:**  
- Xiang, Z. & Li, X. "Fusion of Transformer and ML-CNN-BiLSTM for Network Intrusion Detection." EURASIP Journal on Wireless Communications and Networking, 2023.
- Vaswani, A., et al. "Attention Is All You Need." NeurIPS, 2017.
- Goodfellow, I., et al. "Generative Adversarial Networks." NeurIPS, 2014[1].

**Related GitHub Repositories:**
- [A Hybrid CNN-LSTM Approach for Intelligent Cyber Intrusion Detection System](https://github.com/Aditya-Katkuri/A-Hybrid-CNN-LSTM-Approach-for-Intelligent-Cyber-Intrusion-Detection-System)
- [GAN-CNN-BiLSTM Intrusion Detection](https://thesai.org/Downloads/Volume14No5/Paper_54-A_Method_for_Network_Intrusion_Detection.pdf)
- [DCNNBiLSTM: An Efficient Hybrid Deep Learning-Based Intrusion Detection System](https://github.com/vanlalruata/DCNNBiLSTM-An-Efficient-Hybrid-Deep-Learning-Based-Intrusion-Detection-System)
---
