"""
Question 2: Regression - Superconductivity Dataset
Dataset: https://archive.uci.edu/dataset/464/superconductivty+data
Models: Linear Regression (GD), Polynomial Regression, Ridge, Lasso
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

os.makedirs("outputs/q2", exist_ok=True)

print("=" * 70)
print("QUESTION 2: REGRESSION - SUPERCONDUCTIVITY DATASET")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1. LOAD DATASET
# -----------------------------------------------------------------------------
print("\n[1/6] Loading Superconductivity dataset from UCI...")
superc = fetch_ucirepo(id=464)

X = superc.data.features
y = superc.data.targets

# Flatten target if needed
if hasattr(y, 'values'):
    y = y.values.ravel()

print(f"  Features shape : {X.shape}")
print(f"  Target shape   : {y.shape}")
print(f"  Target (critical temp) range: [{y.min():.2f}, {y.max():.2f}] K")

# Drop NaN
mask = pd.DataFrame(X).notnull().all(axis=1).values
X = pd.DataFrame(X)[mask].values
y = y[mask]
print(f"  After NaN drop : {X.shape}")

# -----------------------------------------------------------------------------
# 2. TRAIN-TEST SPLIT (80/20)
# -----------------------------------------------------------------------------
print("\n[2/6] Splitting 80% train / 20% test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

scaler_X = StandardScaler()
X_train_sc = scaler_X.fit_transform(X_train)
X_test_sc  = scaler_X.transform(X_test)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

def regression_metrics(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    return mse, rmse, r2

def print_metrics(name, y_tr_pred, y_te_pred, y_tr, y_te):
    tr_mse, tr_rmse, tr_r2 = regression_metrics(y_tr, y_tr_pred)
    te_mse, te_rmse, te_r2 = regression_metrics(y_te, y_te_pred)
    print(f"\n  {name}")
    print(f"    {'Metric':<8} {'Train':>10} {'Test':>10}")
    print(f"    {'MSE':<8} {tr_mse:>10.4f} {te_mse:>10.4f}")
    print(f"    {'RMSE':<8} {tr_rmse:>10.4f} {te_rmse:>10.4f}")
    print(f"    {'R2':<8} {tr_r2:>10.4f} {te_r2:>10.4f}")
    return tr_mse, tr_rmse, tr_r2, te_mse, te_rmse, te_r2

results = {}

# -----------------------------------------------------------------------------
# 3a. LINEAR REGRESSION WITH GRADIENT DESCENT (from scratch)
# -----------------------------------------------------------------------------
print("\n[3/6] Training regression models...")
print("\n  -- Linear Regression with Gradient Descent (from scratch) --")

class LinearRegressionGD:
    """Batch Gradient Descent Linear Regression."""
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.costs = []

    def fit(self, X, y):
        n, p = X.shape
        self.theta = np.zeros(p + 1)
        X_b = np.c_[np.ones(n), X]
        for i in range(self.n_iters):
            y_hat = X_b @ self.theta
            residuals = y_hat - y
            grad = (2 / n) * X_b.T @ residuals
            self.theta -= self.lr * grad
            self.costs.append(np.mean(residuals ** 2))
        return self

    def predict(self, X):
        X_b = np.c_[np.ones(len(X)), X]
        return X_b @ self.theta

gd_model = LinearRegressionGD(lr=0.01, n_iters=1000)
gd_model.fit(X_train_sc, y_train)
y_train_pred_gd = gd_model.predict(X_train_sc)
y_test_pred_gd  = gd_model.predict(X_test_sc)

m = print_metrics("Linear Regression (GD, from scratch)",
                  y_train_pred_gd, y_test_pred_gd, y_train, y_test)
results["Linear GD"] = {
    "Train MSE": m[0], "Train RMSE": m[1], "Train R2": m[2],
    "Test MSE":  m[3], "Test RMSE":  m[4], "Test R2":  m[5]
}

# Cost convergence plot
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(gd_model.costs, color='steelblue', lw=2)
ax.set_xlabel('Iteration'); ax.set_ylabel('MSE Loss')
ax.set_title('Linear Regression GD: Cost Convergence', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/q2/gd_convergence.png', dpi=150)
plt.close()
print("  Saved: outputs/q2/gd_convergence.png")

# -----------------------------------------------------------------------------
# 3b. POLYNOMIAL REGRESSION (overfitting / underfitting analysis)
# -----------------------------------------------------------------------------
print("\n  -- Polynomial Regression --")

# Use PCA to reduce to 10 dims first (to keep poly feasible)
from sklearn.decomposition import PCA
pca = PCA(n_components=10, random_state=42)
X_tr_pca = pca.fit_transform(X_train_sc)
X_te_pca  = pca.transform(X_test_sc)

degrees = [1, 2, 3, 4, 5]
tr_mses_poly, te_mses_poly, tr_r2s_poly, te_r2s_poly = [], [], [], []

for deg in degrees:
    pipe = Pipeline([
        ('poly',  PolynomialFeatures(degree=deg, include_bias=False)),
        ('scale', StandardScaler()),
        ('lr',    LinearRegression())
    ])
    pipe.fit(X_tr_pca, y_train)
    tr_p = pipe.predict(X_tr_pca)
    te_p = pipe.predict(X_te_pca)
    tr_mse, _, tr_r2 = regression_metrics(y_train, tr_p)
    te_mse, _, te_r2 = regression_metrics(y_test, te_p)
    tr_mses_poly.append(tr_mse); te_mses_poly.append(te_mse)
    tr_r2s_poly.append(tr_r2);   te_r2s_poly.append(te_r2)
    print(f"    Degree {deg}: Train MSE={tr_mse:.2f}, Test MSE={te_mse:.2f}, "
          f"Train R2={tr_r2:.4f}, Test R2={te_r2:.4f}")

# Overfitting/Underfitting plot
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(degrees, tr_mses_poly, 'o-', label='Train MSE', color='steelblue')
axes[0].plot(degrees, te_mses_poly, 's-', label='Test MSE',  color='orange')
axes[0].set_xlabel('Polynomial Degree'); axes[0].set_ylabel('MSE')
axes[0].set_title('Polynomial Regression: Overfitting Analysis')
axes[0].legend()

axes[1].plot(degrees, tr_r2s_poly, 'o-', label='Train R2', color='steelblue')
axes[1].plot(degrees, te_r2s_poly, 's-', label='Test R2',  color='orange')
axes[1].set_xlabel('Polynomial Degree'); axes[1].set_ylabel('R2')
axes[1].set_title('Polynomial Regression: R2 vs Degree')
axes[1].legend()
plt.tight_layout()
plt.savefig('outputs/q2/poly_overfitting.png', dpi=150)
plt.close()
print("  Saved: outputs/q2/poly_overfitting.png")

# Best polynomial (degree 2 balances bias-variance)
best_deg = degrees[np.argmax(te_r2s_poly)]
print(f"  Best degree (by test R2): {best_deg}")
best_poly = Pipeline([
    ('poly',  PolynomialFeatures(degree=best_deg, include_bias=False)),
    ('scale', StandardScaler()),
    ('lr',    LinearRegression())
])
best_poly.fit(X_tr_pca, y_train)
y_tr_pred_poly = best_poly.predict(X_tr_pca)
y_te_pred_poly = best_poly.predict(X_te_pca)

m = print_metrics(f"Polynomial Regression (deg={best_deg})",
                  y_tr_pred_poly, y_te_pred_poly, y_train, y_test)
results["Polynomial (deg=%d)" % best_deg] = {
    "Train MSE": m[0], "Train RMSE": m[1], "Train R2": m[2],
    "Test MSE":  m[3], "Test RMSE":  m[4], "Test R2":  m[5]
}

# -----------------------------------------------------------------------------
# 3c. RIDGE REGRESSION (with CV for alpha)
# -----------------------------------------------------------------------------
print("\n  -- Ridge Regression (CV for alpha) --")

alphas = [0.001, 0.01, 0.1, 1, 10, 50, 100, 500, 1000]
ridge_cv_scores = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    scores = -cross_val_score(ridge, X_train_sc, y_train,
                              cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    ridge_cv_scores.append(scores.mean())
    print(f"    alpha={alpha:<8} CV MSE={scores.mean():.4f} +/- {scores.std():.4f}")

best_alpha_ridge = alphas[np.argmin(ridge_cv_scores)]
print(f"  Best Ridge alpha (CV): {best_alpha_ridge}")

ridge_best = Ridge(alpha=best_alpha_ridge)
ridge_best.fit(X_train_sc, y_train)
y_tr_pred_ridge = ridge_best.predict(X_train_sc)
y_te_pred_ridge = ridge_best.predict(X_test_sc)

m = print_metrics(f"Ridge Regression (alpha={best_alpha_ridge})",
                  y_tr_pred_ridge, y_te_pred_ridge, y_train, y_test)
results["Ridge (alpha=%s)" % best_alpha_ridge] = {
    "Train MSE": m[0], "Train RMSE": m[1], "Train R2": m[2],
    "Test MSE":  m[3], "Test RMSE":  m[4], "Test R2":  m[5]
}

# Ridge alpha vs CV MSE plot
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogx(alphas, ridge_cv_scores, 'o-', color='steelblue', lw=2)
ax.axvline(best_alpha_ridge, color='red', linestyle='--', label=f'Best alpha={best_alpha_ridge}')
ax.set_xlabel('Alpha (log scale)'); ax.set_ylabel('CV MSE')
ax.set_title('Ridge: Cross-Validation MSE vs Alpha')
ax.legend(); plt.tight_layout()
plt.savefig('outputs/q2/ridge_cv_alpha.png', dpi=150)
plt.close()
print("  Saved: outputs/q2/ridge_cv_alpha.png")

# -----------------------------------------------------------------------------
# 3d. LASSO REGRESSION (with CV for alpha)
# -----------------------------------------------------------------------------
print("\n  -- Lasso Regression (CV for alpha) --")

lasso_alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
lasso_cv_scores = []
for alpha in lasso_alphas:
    lasso = Lasso(alpha=alpha, max_iter=2000, tol=1e-3)
    scores = -cross_val_score(lasso, X_train_sc, y_train,
                              cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    lasso_cv_scores.append(scores.mean())
    print(f"    alpha={alpha:<8} CV MSE={scores.mean():.4f} +/- {scores.std():.4f}")

best_alpha_lasso = lasso_alphas[np.argmin(lasso_cv_scores)]
print(f"  Best Lasso alpha (CV): {best_alpha_lasso}")

lasso_best = Lasso(alpha=best_alpha_lasso, max_iter=2000, tol=1e-3)
lasso_best.fit(X_train_sc, y_train)
y_tr_pred_lasso = lasso_best.predict(X_train_sc)
y_te_pred_lasso = lasso_best.predict(X_test_sc)

m = print_metrics(f"Lasso Regression (alpha={best_alpha_lasso})",
                  y_tr_pred_lasso, y_te_pred_lasso, y_train, y_test)
results["Lasso (alpha=%s)" % best_alpha_lasso] = {
    "Train MSE": m[0], "Train RMSE": m[1], "Train R2": m[2],
    "Test MSE":  m[3], "Test RMSE":  m[4], "Test R2":  m[5]
}

n_nonzero = np.sum(np.abs(lasso_best.coef_) > 1e-6)
print(f"  Lasso non-zero features: {n_nonzero} / {X_train_sc.shape[1]}")

# Lasso alpha vs CV MSE plot
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogx(lasso_alphas, lasso_cv_scores, 'o-', color='green', lw=2)
ax.axvline(best_alpha_lasso, color='red', linestyle='--', label=f'Best alpha={best_alpha_lasso}')
ax.set_xlabel('Alpha (log scale)'); ax.set_ylabel('CV MSE')
ax.set_title('Lasso: Cross-Validation MSE vs Alpha')
ax.legend(); plt.tight_layout()
plt.savefig('outputs/q2/lasso_cv_alpha.png', dpi=150)
plt.close()
print("  Saved: outputs/q2/lasso_cv_alpha.png")

# -----------------------------------------------------------------------------
# 4. ACTUAL vs PREDICTED PLOTS
# -----------------------------------------------------------------------------
print("\n[4/6] Actual vs Predicted scatter plots...")

preds = {
    "Linear GD":       y_test_pred_gd,
    "Polynomial":      y_te_pred_poly,
    "Ridge":           y_te_pred_ridge,
    "Lasso":           y_te_pred_lasso,
}
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()
colors = ['steelblue', 'orange', 'green', 'purple']
for ax, (name, preds_), color in zip(axes, preds.items(), colors):
    ax.scatter(y_test, preds_, alpha=0.3, s=10, color=color)
    mn, mx = min(y_test.min(), preds_.min()), max(y_test.max(), preds_.max())
    ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5)
    r2 = r2_score(y_test, preds_)
    ax.set_title(f'{name} (R2={r2:.4f})', fontweight='bold')
    ax.set_xlabel('Actual Tc (K)'); ax.set_ylabel('Predicted Tc (K)')
plt.suptitle('Actual vs Predicted Critical Temperature', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/q2/actual_vs_predicted.png', dpi=150)
plt.close()
print("  Saved: outputs/q2/actual_vs_predicted.png")

# -----------------------------------------------------------------------------
# 5. RESIDUAL PLOTS
# -----------------------------------------------------------------------------
print("\n[5/6] Residual plots...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()
for ax, (name, preds_), color in zip(axes, preds.items(), colors):
    residuals = y_test - preds_
    ax.scatter(preds_, residuals, alpha=0.3, s=10, color=color)
    ax.axhline(0, color='red', linestyle='--', lw=1.5)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Residuals')
    ax.set_title(f'{name} Residuals')
plt.suptitle('Residual Plots', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/q2/residuals.png', dpi=150)
plt.close()
print("  Saved: outputs/q2/residuals.png")

# -----------------------------------------------------------------------------
# 6. COMPARATIVE SUMMARY
# -----------------------------------------------------------------------------
print("\n[6/6] Final Results Summary")
print("=" * 70)
summary = pd.DataFrame(results).T
summary.columns = ["Train MSE", "Train RMSE", "Train R2", "Test MSE", "Test RMSE", "Test R2"]
print(summary.round(4).to_string())

# Bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics_to_plot = ["Test MSE", "Test RMSE", "Test R2"]
model_names = list(results.keys())
colors_bar = ['steelblue', 'orange', 'green', 'purple']

for ax, metric in zip(axes, metrics_to_plot):
    vals = [results[m][metric] for m in model_names]
    bars = ax.bar(model_names, vals, color=colors_bar)
    ax.set_title(metric, fontweight='bold')
    ax.set_xticklabels(model_names, rotation=20, ha='right', fontsize=9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
plt.suptitle('Regression Models Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/q2/regression_comparison.png', dpi=150)
plt.close()
print("  Saved: outputs/q2/regression_comparison.png")

best_model = min(results.items(), key=lambda x: x[1]["Test MSE"])
print(f"\n  ? Best model by Test MSE: {best_model[0]} (MSE={best_model[1]['Test MSE']:.4f})")
print("\nDone! All outputs saved to outputs/q2/")
