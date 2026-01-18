# Dimensionality Reduction - PCA

This folder contains implementations and projects related to **Principal Component Analysis (PCA)**, a fundamental dimensionality reduction technique.

## 📚 What is PCA?

Principal Component Analysis is an unsupervised learning algorithm that:
- Reduces the number of features while preserving variance in the data
- Identifies the most important directions of variation
- Transforms data to a new coordinate system where the greatest variance lies on the first coordinate (PC1), the second greatest on the second (PC2), etc.

## 📁 Project Structure

### PCA Implementations
- **`using sklearn.ipynb`** - Quick PCA implementation using scikit-learn
- **`pca without sklearn lib.ipynb`** - Manual implementation from scratch
- **`pca without using sklearn.ipynb`** - Alternative from-scratch approach
- **`pixel project manipulation.ipynb`** - Practical exercises with pixel data

### Reference Project
- **`Reference project/digit recognizer/`** - Real-world application using PCA for handwritten digit recognition
  - `pca.ipynb` - Complete PCA implementation for digit classification
  - `train.csv`, `test.csv` - MNIST-like dataset
  - `sample_submission.csv` - Example output format

## 🎯 Key Concepts

### 1. **Covariance Matrix**
Measures how features vary together
```python
Cov = (X - mean(X))^T * (X - mean(X)) / (n-1)
```

### 2. **Eigenvalues and Eigenvectors**
- Eigenvectors: directions of maximum variance (principal components)
- Eigenvalues: magnitude of variance in each direction

### 3. **Variance Explained**
The proportion of variance captured by each principal component
```python
variance_explained = eigenvalue / sum(all_eigenvalues)
```

### 4. **Reconstruction**
Transforming reduced data back to original space

## 🚀 How to Use

### For Beginners
1. Start with `using sklearn.ipynb` to understand PCA basics
2. Examine the visualizations and learn how dimensions are reduced
3. Compare results with different numbers of components

### For Intermediate Learners
1. Read `pca without sklearn lib.ipynb` to understand the mathematical implementation
2. Manually compute covariance matrix and eigenvalues
3. Experiment with different datasets

### For Advanced Learners
1. Implement PCA from scratch following `pca without using sklearn.ipynb`
2. Optimize for large datasets
3. Apply to the digit recognizer reference project

## 📊 Step-by-Step Algorithm

```
1. Standardize the data (mean=0, std=1)
2. Calculate covariance matrix
3. Find eigenvalues and eigenvectors
4. Sort by eigenvalues in descending order
5. Select top k eigenvectors
6. Transform data: X_reduced = X @ eigenvectors[:, :k]
7. (Optional) Reconstruct: X_reconstructed = X_reduced @ eigenvectors[:, :k].T
```

## 💡 Practical Applications

- **Image Compression:** Reduce image dimensions while maintaining quality
- **Digit Recognition:** Classify handwritten digits with fewer features
- **Data Visualization:** Reduce to 2D/3D for visualization
- **Noise Reduction:** Remove irrelevant features
- **Feature Engineering:** Create new meaningful features

## 📈 When to Use PCA

✅ **Use PCA when:**
- You have high-dimensional data
- You want to visualize data in 2D/3D
- You need to reduce computation time
- Features are highly correlated
- You want to understand variance in your data

❌ **Don't use PCA when:**
- You have very few features (curse of dimensionality doesn't apply)
- You need interpretable features (PCA creates new abstract features)
- Your data is not linearly distributed (use kernel PCA instead)

## 🔗 Resources

- [Scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Understanding PCA](https://towardsdatascience.com/pca-explained-visually-with-zero-math-1c092f7f1d25)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

---

**Last Updated:** January 2026

