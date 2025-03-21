# MNIST Classification Using MLE, PCA, FDA, and Discriminant Analysis

## Project Overview

This project focuses on building a classification pipeline for the MNIST dataset, specifically for the handwritten digits `0`, `1`, and `2`. The objective is to implement key machine learning concepts such as **Maximum Likelihood Estimation (MLE)**, **Principal Component Analysis (PCA)**, **Fisher’s Discriminant Analysis (FDA)**, and **Discriminant Analysis (LDA/QDA)** for dimensionality reduction and classification.

---

## Dataset Information

The dataset used in this project is the MNIST dataset, which consists of handwritten digit images (0-9). You can access the dataset [here](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).

For this task:
- **Train Set**: 100 random samples for each class (0, 1, 2) – total of 300 samples.
- **Test Set**: 100 random samples for each class (0, 1, 2) – total of 300 samples.

---

## Task Description

### Key Steps:
1. **Data Preprocessing**:
   - Filter classes `0`, `1`, and `2`.
   - Convert images into feature vectors by stacking columns.
   - Normalize features to the range [0, 1].

2. **Compute MLE Estimates**:
   - Estimate the mean (`μc`) and covariance matrix (`Σc`) for each digit class.
   - Assume a **multivariate Gaussian distribution** for the data.

3. **Dimensionality Reduction Using PCA**:
   - Apply PCA to retain 95% variance.
   - Perform dimensionality reduction and transform the data accordingly.

4. **Class Projection Using Fisher’s Discriminant Analysis (FDA)**:
   - Compute between-class and within-class scatter matrices.
   - Solve a **generalized eigenvalue problem** to find the optimal projection.

5. **Classification Using LDA/QDA**:
   - Train LDA and QDA classifiers on the transformed data.
   - Evaluate test set accuracy for both classifiers.

6. **Performance Analysis**:
   - Compare classification accuracy with and without PCA.
   - Analyze performance under different PCA configurations:
     - Retaining 90% variance.
     - Using only the first two principal components.

7. **Visualization**:
   - Generate 2D visualizations of the transformed feature spaces (from PCA and FDA).

---

## Accuracy Metrics

Accuracy is measured as:

$$\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}$$

### Special Cases:
- **Perfect Accuracy**: 100% predictions correct.
- **Worst Accuracy**: 0% predictions correct.

**Note**: For imbalanced datasets, metrics like Precision, Recall, and F1-score may also be relevant.

---

## Results to Report

Your final report should include:
- Classification accuracy for both LDA and QDA.
- Impact of PCA on classification performance.
- Analysis of performance under varying PCA configurations.
- 2D visualizations of the feature space for PCA and FDA-transformed data.

---

## How to Run the Project

### Prerequisites
- Python 3.7+
- Libraries: `numpy`, `scikit-learn`, `matplotlib`, `pandas`.

### Steps:
1. Download the MNIST dataset from the link provided.
2. Preprocess the data as described above.
3. Implement MLE, PCA, FDA, and LDA/QDA in sequence.
4. Run the classifiers and evaluate performance.
5. Visualize results using `matplotlib`.

### File Structure:
- `data/`: Contains the MNIST dataset (to be downloaded).
- `notebooks/`: Jupyter notebooks for each step of the implementation.
- `visualizations/`: Stores plots for transformed feature spaces.

---

## Examples of Visualization

### PCA Transformed Data
- Plot the 2D PCA projection with different colors/markers for each class.
- ![image](https://github.com/user-attachments/assets/97f799b7-60ce-47ad-a5b8-f090abe8a63f)


### FDA Transformed Data
- Plot the 2D FDA projection with class separability.
- ![image](https://github.com/user-attachments/assets/57d14eeb-c2bf-4954-add9-11a2862d0840)


---

## Future Scope

- Extend the pipeline to additional digit classes (3-9).
- Evaluate performance on other datasets.
- Experiment with other dimensionality reduction methods like **t-SNE** or **UMAP**.

---

