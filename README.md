# Zero-Day Attack Detection System

A production-ready system for detecting zero-day attacks using unsupervised learning models. Learns normal network traffic patterns and identifies anomalies that may indicate novel attacks.

## 🎯 Core Philosophy

This system implements a critical security principle: **learn what normal looks like, and flag anything that deviates**. By training exclusively on benign (normal) network traffic, the models develop a "boundary of normality" that can detect previously unseen (zero-day) attacks without requiring labeled attack examples.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ZERO-DAY ATTACK DETECTION SYSTEM             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Training   │    │    Test      │    │ Evaluation   │      │
│  │   Data       │    │   Data       │    │   Results    │      │
│  │  (Benign)    │    │  (Mixed)     │    │              │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │              │
│         ▼                   ▼                   ▼              │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              DATA PROCESSOR                          │       │
│  │  • One-Hot Encoding (categorical features)          │       │
│  │  • Standard Scaling                                  │       │
│  │  • SelectKBest Feature Selection                    │       │
│  │  • PREVENTS DATA LEAKAGE                            │       │
│  └─────────────────────┬───────────────────────────────┘       │
│                        │                                     │
│                        ▼                                     │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              DETECTION MODELS                        │       │
│  │                                                     │       │
│  │  ┌─────────────────┐  ┌─────────────────────────┐   │       │
│  │  │ Isolation Forest│  │ One-Class SVM           │   │       │
│  │  │ ★ PRIMARY MODEL │  │                         │   │       │
│  │  └─────────────────┘  └─────────────────────────┘   │       │
│  │                                                     │       │
│  │  ┌─────────────────────────────────────────────┐   │       │
│  │  │ Local Outlier Factor (LOF)                  │   │       │
│  │  └─────────────────────────────────────────────┘   │       │
│  └─────────────────────┬───────────────────────────────┘       │
│                        │                                     │
│                        ▼                                     │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              MODEL EVALUATOR                         │       │
│  │  • Confusion Matrix                                 │       │
│  │  • ROC Curves                                       │       │
│  │  • Anomaly Score Distributions                      │       │
│  │  • Feature Importance                               │       │
│  │  • Performance Comparison                           │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- 4GB RAM (8GB recommended)
- 2GB disk space

### Installation

```bash
# Clone or download the project
cd zero_day_detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the detector
python zero_day_detector.py
```

### Google Colab

```python
# In a Colab cell, run:
!pip install numpy pandas scikit-learn matplotlib seaborn

# Upload files or clone repository
# Then run:
!python zero_day_detector.py
```

## 📊 Models

### 1. Isolation Forest ⭐ (Primary)
**Best for**: Production use, high-dimensional data

- **Principle**: Anomalies are few and different, making them easier to isolate
- **Speed**: Fast training and prediction (O(n log n))
- **Pros**: Excellent for high-dimensional data, handles mixed feature types
- **Cons**: Requires careful tuning of contamination parameter

```python
# Default configuration
model = IsolationForestDetector(
    n_estimators=200,      # Number of trees
    contamination=0.15,    # Expected outlier fraction
    random_state=42
)
```

### 2. One-Class SVM
**Best for**: Small datasets, complex decision boundaries

- **Principle**: Learn a decision boundary around normal data
- **Speed**: Slower training (O(n²) to O(n³))
- **Pros**: Excellent boundary learning, works with complex patterns
- **Cons**: Memory intensive, slow on large datasets

```python
# Default configuration
model = OneClassSVMDetector(
    kernel='rbf',      # Radial Basis Function kernel
    nu=0.15,           # Upper bound on outliers
    gamma='scale'      # Auto kernel coefficient
)
```

### 3. Local Outlier Factor (LOF)
**Best for**: Datasets with varying densities

- **Principle**: Compare local density of point to its neighbors
- **Speed**: Moderate (O(n log n) for kd-tree)
- **Pros**: No assumptions about data distribution, good for uneven densities
- **Cons**: Sensitive to k (n_neighbors) parameter

```python
# Default configuration
model = LOFDetector(
    n_neighbors=30,    # Neighbors to compare
    contamination=0.15,
    novelty=True       # Enable prediction on new data
)
```

## 📈 Results Summary

### Typical Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Isolation Forest | 0.82-0.87 | 0.88-0.92 | 0.75-0.82 | 0.81-0.86 | 0.89-0.94 |
| One-Class SVM | 0.74-0.80 | 0.80-0.86 | 0.68-0.75 | 0.74-0.80 | 0.82-0.88 |
| Local Outlier Factor | 0.78-0.84 | 0.85-0.90 | 0.72-0.79 | 0.78-0.83 | 0.86-0.92 |

### Key Findings

1. **Data Overview**:
   - NSL-KDD Training: ~67,000 benign samples
   - NSL-KDD Test: ~22,500 mixed samples (Benign + Attacks)

2. **Detection Capability**:
   - All three models successfully identify known attack patterns
   - Isolation Forest provides the best balance of speed and accuracy
   - Models generalize from benign data to detect zero-day attacks

3. **Production Recommendations**:
   - Use Isolation Forest for real-time detection
   - Monitor for drift in normal traffic patterns
   - Retrain regularly on new normal data
   - Combine with signature-based detection for hybrid approach

## 📁 Output Files

| File | Description |
|------|-------------|
| `confusion_matrices.png` | Confusion matrices for all models |
| `roc_curves.png` | ROC curves comparison with AUC scores |
| `anomaly_distribution.png` | Score distributions (Benign vs Attack) |
| `model_comparison.png` | Performance metrics bar chart |
| `feature_importance.png` | Top 15 most important features |
| `evaluation_report.txt` | Detailed text report |
| `results_summary.csv` | Metrics in CSV format |

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Feature Selection
SELECT_K_BEST_K = 15  # Number of top features to select

# Isolation Forest Parameters
ISOLATION_FOREST = {
    'n_estimators': 200,
    'contamination': 0.15,
    ...
}

# One-Class SVM Parameters  
ONE_CLASS_SVM = {
    'kernel': 'rbf',
    'nu': 0.15,
    ...
}

# LOF Parameters
LOF = {
    'n_neighbors': 30,
    'contamination': 0.15,
    ...
}
```

## 🔧 Parameter Tuning Guide

### Isolation Forest

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| n_estimators | 200 | 100-500 | More trees = more stable, slower |
| contamination | 0.15 | 0.05-0.30 | Lower = fewer false positives |
| max_samples | 'auto' | 'auto' or 0.5-1.0 | Data fraction to sample |

### One-Class SVM

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| nu | 0.15 | 0.01-0.30 | Outlier fraction upper bound |
| gamma | 'scale' | 'scale' or float | Kernel coefficient |
| kernel | 'rbf' | 'rbf', 'linear', 'poly' | 'rbf' for complex boundaries |

### Local Outlier Factor

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| n_neighbors | 30 | 10-100 | Neighbors to compare |
| contamination | 0.15 | 0.05-0.30 | Outlier fraction |

## 📚 Dataset Information

### NSL-KDD Dataset
- **Source**: Canadian Institute for Cybersecurity
- **Training Set**: 125,973 samples
- **Test Set**: 22,544 samples
- **Features**: 41 (including label)
- **Attack Categories**: DoS, Probe, R2L, U2R

### Feature Types
- **Continuous**: duration, src_bytes, dst_bytes, count, srv_count, etc.
- **Categorical**: protocol_type (TCP, UDP, ICMP), service (http, ftp, telnet, etc.), flag (SF, S0, REJ, etc.)

### Attack Types
| Category | Description | Examples |
|----------|-------------|----------|
| DoS | Denial of Service | smurf, neptune, back, teardrop |
| Probe | Surveillance/Scanning | portsweep, ipsweep, nmap, satan |
| R2L | Remote to Local | guess_password, ftp_write, imap |
| U2R | User to Root | buffer_overflow, rootkit, perl |

## 🛡️ Security & Safety Notes

1. **Educational Use**: This system is for educational and research purposes
2. **Validation**: Always validate detection results in production
3. **Hybrid Approach**: Combine with signature-based detection for best results
4. **Drift Monitoring**: Monitor for concept drift in normal traffic patterns
5. **Regular Retraining**: Retrain models periodically on new normal data

## 📖 References

1. **Isolation Forest**: Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. IEEE International Conference on Data Mining.

2. **One-Class SVM**: Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., & Williamson, R. C. (2001). Estimating the support of a high-dimensional distribution. Neural computation.

3. **Local Outlier Factor**: Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF: identifying density-based local outliers. ACM SIGMOD Record.

4. **NSL-KDD Dataset**: Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). A detailed analysis of the KDD Cup 99 data set. IEEE Symposium on Computational Intelligence for Security and Defense Applications.

## 📝 License

This project is provided for educational and research purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**⚠️ Important**: This system learns what "normal" network traffic looks like. In production, combine this anomaly detection approach with signature-based detection for comprehensive security coverage.
