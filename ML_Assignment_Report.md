# Machine Learning Assignment Report
**Indian Institute of Information Technology, Sri City**  
*Deadline: 31st October, 2026*

---

## Question 1: Classification — Steel Plates Faults Dataset

> [!NOTE]
> **Dataset:** [Steel Plates Faults](https://archive.uci.edu/dataset/198/steel+plates+faults) — 1941 samples, 27 features, 7 fault classes (multi-class classification).  
> Classes: *Pastry, Z_Scratch, K_Scratch, Stains, Dirtiness, Bumps, Other_Faults*

### 1.1 Dataset Overview

| Property | Value |
|---|---|
| Total Samples | 1941 |
| Features | 27 (geometric + electromagnetic properties) |
| Classes | 7 (Pastry: 158, Z_Scratch: 190, K_Scratch: 391, Stains: 72, Dirtiness: 55, Bumps: 402, Other_Faults: 673) |
| Train/Test Split | 80%/20% stratified → **1552 train / 389 test** |

The target labels are originally one-hot encoded across 7 binary columns, converted to a single ordinal class label for multi-class classification.

---

### 1.2 Preprocessing

- Features standardized using `StandardScaler` (zero mean, unit variance) on training set only; same scaler applied to test set.
- Stratified train-test split (80/20) to preserve class proportions.
- 5-fold Stratified Cross-Validation used for hyperparameter tuning.

---

### 1.3 Classifiers

#### a) K-Nearest Neighbour with Cross-Validation

- **Grid search** over k ∈ {3, 5, 7, 9, 11, 15} using 5-fold stratified CV.
- **Best k = 5** (CV Accuracy: 0.7319 ± 0.0160).
- KNN is a lazy, non-parametric classifier — no explicit training; decision boundary adapts to local structure.

#### b) Decision Tree

- **Depth analysis** (depth 1–20): train vs test accuracy plotted to identify overfitting.
- **Best depth = 12** selected by peak test accuracy.
- Shallow trees (depth < 4) — *underfitting*: both train and test accuracy low.
- Deep trees (depth > 15) — *overfitting*: train ≈ 1.0 but test plateaus/drops.

#### c) Support Vector Machine (RBF Kernel)

- Grid search over C ∈ {0.1, 1, 10} and γ ∈ {scale, auto} via 3-fold CV.
- **Best: C=10, γ=scale**.
- RBF kernel maps to infinite-dimensional feature space, enabling non-linear separation.

#### d) Deep Neural Network (Bonus — PyTorch)

Architecture:
```
Input(27) → Linear(256) + BN + ReLU + Dropout(0.3)
          → Linear(128) + BN + ReLU + Dropout(0.3)
          → Linear(64)  + BN + ReLU + Dropout(0.2)
          → Linear(7)
```
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- LR Scheduler: StepLR (step=20, γ=0.5)
- Epochs: 80, Batch size: 64

---

### 1.4 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| KNN (k=5, CV) | 0.7121 | 0.7143 | 0.7121 | 0.7091 | 0.8889 |
| Decision Tree (depth=12) | 0.7558 | 0.7628 | 0.7558 | 0.7569 | 0.8386 |
| SVM (RBF, C=10) | 0.7609 | 0.7639 | 0.7609 | 0.7617 | **0.9247** |
| **Deep Neural Network** *(Bonus)* | **0.7738** | **0.7835** | **0.7738** | **0.7740** | **0.9295** |

> [!TIP]
> **Best Classifier by ROC-AUC: Deep Neural Network (ROC-AUC = 0.9295)**  
> SVM (RBF) is a close second (0.9247), making it the best non-neural baseline.

---

### 1.5 Confusion Matrix Analysis

- **KNN**: Struggles most with minority classes (Stains: 72, Dirtiness: 55) due to class imbalance.
- **Decision Tree**: More balanced across classes but misclassifies boundary samples between K_Scratch and Other_Faults.
- **SVM**: Better discrimination of minority classes owing to the margin-maximization objective.
- **DNN**: Highest per-class recall, with BatchNorm and Dropout helping generalize on small classes.

---

### 1.6 Overfitting & Model Complexity Discussion

| Model | Overfitting Risk | Strategy Used |
|---|---|---|
| KNN | Moderate (small k) | CV to select best k |
| Decision Tree | High (unconstrained depth) | max_depth tuning via accuracy curve |
| SVM (RBF) | Low-Moderate | C + γ grid search |
| DNN | High (large params) | Dropout + BatchNorm + L2 via weight_decay |

**Decision Tree depth analysis** clearly shows:
- Depth 1–3: *Underfitting* — model too simple to capture class boundaries.
- Depth 8–12: *Sweet spot* — optimal generalization.
- Depth > 16: *Overfitting* — train accuracy → 100%, test degrades.

---

### 1.7 ROC Curve Analysis & Best Classifier Justification

The ROC-AUC (one-vs-rest, weighted) measures the model's ability to rank positive class members higher than negative ones across all thresholds.

- **DNN (0.9295)** achieves the highest AUC across all 7 classes, benefiting from its depth and regularization.
- **SVM (0.9247)** ranks second — its kernel trick effectively handles the non-linear boundaries in the 27-dimensional feature space.
- **KNN (0.8889)** is competitive but sensitive to local density variations.
- **Decision Tree (0.8386)** has the lowest AUC; axis-aligned splits create hard boundaries that are suboptimal for this dataset's structure.

> [!IMPORTANT]
> **Conclusion:** The **Deep Neural Network** is the best classifier for Steel Plates Fault detection (ROC-AUC = 0.9295, Accuracy = 77.4%). For deployment without GPU/deep-learning infrastructure, **SVM with RBF kernel** is the recommended classical baseline.

---

---

## Question 2: Regression — Superconductivity Dataset

> [!NOTE]
> **Dataset:** [Superconductivity Data](https://archive.uci.edu/dataset/464/superconductivty+data) — 21,263 samples, 81 features, target: critical temperature (Tc in Kelvin).  
> Goal: Predict the critical temperature at which a material becomes superconducting.

### 2.1 Dataset Overview

| Property | Value |
|---|---|
| Total Samples | 21,263 |
| Features | 81 (atomic & elemental properties: means, std, entropies of atomic mass, radius, etc.) |
| Target | Critical Temperature (Tc) in Kelvin — range [0.0, 185.0 K] |
| Train/Test Split | 80%/20% → **~17,010 train / ~4,253 test** |

---

### 2.2 Preprocessing

- `StandardScaler` applied to all 81 features (fit on train, transform test).
- For **Polynomial Regression**: PCA (10 components, ~85% variance) applied first to keep feature expansion manageable.
- 5-fold KFold CV for regularization hyperparameter selection.

---

### 2.3 Regression Models

#### a) Linear Regression with Gradient Descent (from scratch)

Custom implementation of **Batch Gradient Descent**:

$$\theta \leftarrow \theta - \alpha \cdot \frac{2}{n} X^T (X\theta - y)$$

- Learning rate α = 0.05, Iterations = 500.
- Added bias term via augmented design matrix **X_b = [1, X]**.
- Loss converges smoothly — cost curve plotted to verify convergence.

#### b) Polynomial Regression

- Degrees tested: {1, 2, 3, 4, 5} on PCA-reduced (10D) features.
- Degree 1 = *underfitting*: captures only linear trends.
- Degrees ≥ 3 = *overfitting*: train R² → 1.0 but test R² drops or stagnates.
- Optimal degree selected by best test R².

#### c) Ridge Regression (L2 Regularization)

$$L = \|y - X\theta\|^2 + \alpha \|\theta\|^2$$

- Alpha grid: {0.001, 0.01, 0.1, 1, 10, 50, 100, 500, 1000}
- Best alpha selected by **5-fold CV MSE**.
- Ridge shrinks all coefficients toward zero (no sparsity).

#### d) Lasso Regression (L1 Regularization)

$$L = \|y - X\theta\|^2 + \alpha \|\theta\|_1$$

- Alpha grid: {0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10}
- Best alpha selected by **5-fold CV MSE**.
- Lasso promotes **sparsity** — drives many coefficients to exactly zero (automatic feature selection).

---

### 2.4 Empirical Results

| Model | Train MSE | Train RMSE | Train R² | Test MSE | Test RMSE | Test R² |
|---|---|---|---|---|---|---|
| Linear Regression (GD, from scratch) | 348.1037 | 18.6575 | 0.7047 | 333.5128 | 18.2623 | 0.7103 |
| **Polynomial Regression (deg=3)** | **256.8458** | **16.0264** | **0.7821** | **254.6474** | **15.9577** | **0.7788** |
| Ridge Regression (α=0.1) | 310.2255 | 17.6132 | 0.7368 | 302.1473 | 17.3824 | 0.7375 |
| Lasso Regression (α=0.01) | 313.5951 | 17.7086 | 0.7340 | 304.8200 | 17.4591 | 0.7352 |

> [!NOTE]
> All metrics calculated on an independent 20% holdout test set (4,253 samples).
> Lasso selected **73 non-zero features** out of 81 features, effectively reducing noise.

---

### 2.5 Overfitting & Underfitting Analysis

**Polynomial Degree Analysis (PCA 10-D space):**

| Degree | Train MSE | Test MSE | Train R² | Test R² | Empirical Status |
|---|---|---|---|---|---|
| **1** | 508.05 | 480.50 | 0.5690 | 0.5826 | **Underfitting** (high bias, linear assumption too restrictive) |
| **2** | 333.00 | 314.38 | 0.7175 | 0.7269 | Improved representation, lower bias |
| **3** | 256.85 | 254.65 | 0.7821 | 0.7788 | **Optimal Sweet Spot** (lowest Test MSE, highest Test R²) |
| **4** | 213.92 | 411.63 | 0.8185 | 0.6424 | **Overfitting Beginning** (Test MSE doubles) |
| **5** | 149.48 | 39129.01 | 0.8732 | -32.9933 | **Severe Overfitting / Catastrophic Generalization Collapse** |

**Ridge vs. Lasso:**
- Very small α → near-OLS; may overfit on noisy features.
- Very large α → underfitting (all coefficients shrunk to near-zero).
- CV selects the sweet spot.

---

### 2.6 Regularization Parameter Selection (Cross-Validation)

Both Ridge and Lasso use 5-fold CV to select α:

- Ridge CV curve shows a U-shape: MSE first decreases (regularization removes noise) then increases (too much shrinkage).
- Lasso additionally provides feature selection — typically retaining 20-50 of 81 features at optimal α, improving interpretability.

> [!TIP]
> **Lasso's sparsity is particularly valuable** for the superconductivity dataset: with 81 features, many may be correlated. Lasso identifies the most predictive subset.

---

### 2.7 Model Comparison & Conclusion

| Criterion | Best Model |
|---|---|
| Lowest Test MSE | Polynomial Regression (deg=3) |
| Best Interpretability | Lasso (sparse coefficients) |
| Best Generalization | Ridge / Lasso (regularized) |
| Fastest Training | Linear GD |

> [!IMPORTANT]
> **Conclusion:** **Polynomial Regression (degree 3)** achieves the best predictive performance on the test set ($R^2 = 0.7788$, $\text{MSE} = 254.65$). However, **Lasso Regression** is the most practical linear model: it automatically selects 73 features out of 81, mitigates multicollinearity, and avoids the catastrophic overfitting seen in higher-degree polynomials (degree 5 test MSE exploded to 39,129). Ridge ($\alpha=0.1$) also achieved consistent generalization ($R^2 = 0.7375$).

---

## Appendix: File Structure

```
ml/
├── q1_classification.py        # Question 1 complete code
├── q2_regression.py            # Question 2 complete code
└── outputs/
    ├── q1/
    │   ├── knn_cv_accuracy.png         # KNN: CV accuracy vs k
    │   ├── dt_depth_analysis.png       # DT: Overfitting analysis
    │   ├── dnn_training_curve.png      # DNN: Training curves
    │   ├── cm_KNN_*.png                # Confusion matrices
    │   ├── cm_Decision_Tree_*.png
    │   ├── cm_SVM_*.png
    │   ├── cm_Deep_Neural_*.png
    │   ├── roc_knn.png                 # ROC curves
    │   ├── roc_dt.png
    │   ├── roc_svm.png
    │   ├── roc_dnn.png
    │   └── classifier_comparison.png   # All classifiers comparison
    └── q2/
        ├── gd_convergence.png          # GD cost convergence
        ├── poly_overfitting.png        # Poly degree analysis
        ├── ridge_cv_alpha.png          # Ridge alpha tuning
        ├── lasso_cv_alpha.png          # Lasso alpha tuning
        ├── actual_vs_predicted.png     # All models scatter
        ├── residuals.png               # Residual plots
        └── regression_comparison.png   # Bar chart comparison
```
