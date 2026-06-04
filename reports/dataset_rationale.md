# Project Note: Dataset Strategy and Rationale

**Project:** spatial-ihc-feature-lab
**Focus:** ML-First Spatial IHC Feature Engineering

This document outlines the strategic decision to utilize MIBI-TNBC as the primary dataset for spatial feature engineering, while relegating MIHIC to a secondary role for supportive computer vision (CV) tasks. 

## 1. Core Project Philosophy
The primary objective of this project is to build a robust skill set in **computational pathology**, specifically focusing on feature engineering, spatial biology, and interpretable machine learning. The goal is *not* to build a black-box deep learning platform, but rather to demonstrate how to extract meaningful biological insights from multiplex imaging data. Small-to-medium biomedical datasets reward leakage-safe splits, robust handcrafted features, and biological interpretability far more than oversized deep learning experiments.

## 2. Why MIBI-TNBC is the Primary Dataset
The Multiplexed Ion Beam Imaging (MIBI) dataset provides processed multiplex imaging data, including pre-segmented cells, marker expression values, spatial coordinates, and patient/sample-level metadata. 

This tabular/spatial structure is the ideal foundation for the core project goals:
* **Spatial Feature Engineering:** It enables the calculation of complex biological relationships, such as nearest-neighbor distances between immune and tumor cells, radius contact fractions, and co-localization metrics.
* **Graph-Style Networks:** Coordinates allow for the construction of k-nearest-neighbor graphs and cell-type assortativity analyses.
* **Interpretable Classical ML:** Engineered features can be fed into interpretable models (Logistic Regression, Random Forest, XGBoost) to predict spatial phenotypes or outcome proxies, avoiding the opacity of deep neural networks.
* **Industry Relevance:** In modern biotech, once cells and proteins are segmented, the decisive analytical work shifts to feature selection and biological interpretation—skills perfectly suited for the MIBI dataset.

## 3. Why MIHIC is the Secondary Dataset
The Multiplexed Immunohistochemistry (MIHIC) dataset consists primarily of raw IHC image patches and tissue labels. 

While valuable, relying on MIHIC as the primary dataset risks pivoting the project into an endless deep-learning hyperparameter tuning exercise. Instead, MIHIC serves a specific, constrained purpose:
* **Computer Vision Competence:** It provides a sandbox to practice classical image feature extraction (color histograms, GLCM/texture features) without deep learning.
* **Transfer Learning Practice:** It offers enough train/validation/test splits to train a modest CNN (e.g., ResNet18/EfficientNet) for tissue classification, proving baseline deep learning mechanics without chasing state-of-the-art benchmarks.

## 4. Conclusion
By treating MIBI as the core spatial engine and MIHIC as a supporting CV module, the project maintains a clear narrative. It avoids the common pitfall of mixing too many raw datasets and losing the main thread, resulting in a clean, reproducible portfolio asset focused on biological feature engineering.