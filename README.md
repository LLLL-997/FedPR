# FedPR

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
## Table of Contents
- [Introduction](#introduction)
- [Problem & Challenges](#problem--challenges)
- [Proposed Method: FedPR](#proposed-method-fedpr)
- [Key Features](#key-features)
- [Experimental Results](#experimental-results)
- [License](#license)

## Introduction

This repository contains the official implementation of **FedPR**, a novel federated learning-based Point-Of-Interest (POI) rating method. POI rating systems are crucial tools for evaluating locations based on user preferences and satisfaction. However, with the proliferation of diverse POI platforms (e.g., for dining, entertainment, travel), significant discrepancies in ratings for the same POI often arise due to differing functional emphases and user bases. This can misguide users seeking objective evaluations. FedPR aims to provide more objective and reliable multi-platform POI ratings while strictly preserving data privacy.

## Problem & Challenges

Existing POI rating methods primarily focus on single-platform scenarios, lacking effective solutions for cross-platform collaboration. Addressing multi-platform POI rating presents two formidable challenges:
1.  **Data Isolation:** Due to legal regulations and privacy concerns, direct sharing of sensitive user and POI data across platforms is prohibited.
2.  **Data Heterogeneity:** Different platforms emphasize different POI categories, leading to significant rating biases and extreme data imbalance, especially for "weak" (non-dominant) functions where data is sparse and unreliable. This heterogeneity can severely impact model accuracy and generalization across platforms.

## Proposed Method: FedPR

FedPR is designed to overcome these challenges by enabling collaborative training across platforms without compromising data privacy. Our method strategically improves the accuracy of each platform's weak functions through a combination of innovative techniques:

* **Federated Training:** Utilizes common, unlabeled public POI data for joint training, allowing platforms to enrich their weak function data by acquiring external, diverse knowledge.
* **Knowledge Distillation:** Integrates a sophisticated knowledge distillation mechanism during local model updates. This empowers local models to adaptively learn necessary knowledge from the global model, effectively mitigating catastrophic forgetting of dominant local functions while significantly boosting performance on weak functions.

## Experimental Results

Extensive and rigorous experiments conducted on three real-world datasets (CA, IN, NJ) demonstrate the superior effectiveness and robustness of FedPR compared to state-of-the-art baselines. Our comprehensive evaluation includes:
* **Performance Comparisons:** FedPR consistently outperforms baselines in terms of MAE, RMSE, and Accuracy, especially for challenging weak functions.
* **Parameter Sensitivity Analysis:** Detailed studies on key hyperparameters (e.g., knowledge distillation temperature, distillation weight) validate their optimal configurations and contribute to the model's stability.
* **Robustness Analysis:** FedPR exhibits remarkable resilience to varying degrees of data heterogeneity (non-IID distribution) and extreme data sparsity, confirming its reliability in real-world scenarios.

For detailed experimental results and analysis, please refer to our paper.
