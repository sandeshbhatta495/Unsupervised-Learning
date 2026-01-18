# Unsupervised Learning Projects

This repository contains comprehensive implementations and projects for **Unsupervised Learning** algorithms. Each section includes both theoretical understanding and practical implementations with visualizations.

## 📚 Table of Contents

1. **[05a - Dimensionality Reduction](#dimensionality-reduction)** - Principal Component Analysis (PCA)
2. **[05b - Clustering](#clustering)** - K-Means Algorithm
3. **[05c - Anomaly Detection](#anomaly-detection)** - Anomaly Detection Techniques

---

## 🎯 Dimensionality Reduction

**Location:** `05a_DIMENSIONALITY_REDUCTION/`

### PCA (Principal Component Analysis)
PCA is a technique to reduce the dimensionality of data while preserving as much variance as possible.

**Available Resources:**
- `using sklearn.ipynb` - Scikit-learn implementation
- `pca without sklearn lib.ipynb` - From-scratch implementation
- `pixel project manipulation.ipynb` - Practical pixel manipulation examples
- **Reference Project:** Digit Recognizer using PCA

**Key Concepts:**
- Covariance matrix computation
- Eigenvalue decomposition
- Dimensionality reduction
- Data visualization in lower dimensions

---

## 🎯 Clustering

**Location:** `05b_CLUSTERING/K Mean/`

### K-Means Clustering
K-Means is an unsupervised learning algorithm for partitioning data into K clusters.

**Available Resources:**
- `K Mean.md` - Theory and concepts
- **New:** `kmeans_with_sklearn.ipynb` - Scikit-learn implementation with visualization
- **New:** `kmeans_from_scratch.ipynb` - From-scratch implementation with step-by-step visualization

**Key Concepts:**
- Centroid initialization
- Assignment step
- Update step
- Convergence criteria
- Elbow method for optimal K selection

**Applications:**
- Customer segmentation
- Image compression
- Document clustering

---

## 🎯 Anomaly Detection

**Location:** `05c_ANOMALY_DETECTION/`

### Anomaly Detection using Gaussian Distribution
Detect unusual patterns and outliers in data using statistical methods.

**Available Resources:**
- `Anomaly_Detection.md` - Theory and concepts
- **New:** `anomaly_with_sklearn.ipynb` - Using isolation forest and other sklearn methods
- **New:** `anomaly_from_scratch.ipynb` - Gaussian distribution method from scratch with visualization

**Key Concepts:**
- Gaussian distribution parameters
- Probability density estimation
- Anomaly threshold selection
- Performance metrics (precision, recall, F1-score)

**Applications:**
- Fraud detection
- Network intrusion detection
- Sensor failure detection
- Quality control

---

## 📊 Project Structure

```
05_UNSUPERVISED_LEARNING/
├── README.md (this file)
├── 05a_DIMENSIONALITY_REDUCTION/
│   ├── README.md
│   └── PCA/
├── 05b_CLUSTERING/
│   ├── README.md
│   └── K Mean/
└── 05c_ANOMALY_DETECTION/
    ├── README.md
    └── (Project notebooks)
```

---

## 🚀 Getting Started

### Prerequisites
```bash
numpy
pandas
matplotlib
scikit-learn
scipy
```

### Installation
```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

### Running the Notebooks
1. Navigate to the desired project folder
2. Open the Jupyter notebook
3. Run cells sequentially to see implementations and visualizations

---

## 📈 Learning Outcomes

After completing these projects, you will understand:

✅ How unsupervised learning algorithms work without labeled data  
✅ How to reduce dimensions while preserving information  
✅ How to cluster data and interpret results  
✅ How to detect anomalies in datasets  
✅ How to implement algorithms from scratch  
✅ How to use scikit-learn libraries efficiently  
✅ How to visualize high-dimensional data  

---

## 📝 Notes

- All projects include both **library-based** and **from-scratch** implementations
- Each implementation includes **visualizations** to help understand the algorithm
- Reference projects demonstrate real-world applications
- Follow along with the code and experiment with different parameters

---

## 🔗 Useful Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [Machine Learning by Andrew Ng (Coursera)](https://www.coursera.org/learn/machine-learning)

---

**Last Updated:** January 2026

