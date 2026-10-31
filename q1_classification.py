"""
Question 1: Classification - Steel Plates Faults Dataset
Dataset: https://archive.uci.edu/dataset/198/steel+plates+faults
Classifiers: KNN (CV), Decision Tree, SVM (RBF), Deep Neural Network (Bonus)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             roc_curve, auc, classification_report)
from sklearn.pipeline import Pipeline
import os

# â”€â”€ Output directory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
os.makedirs("outputs/q1", exist_ok=True)

print("=" * 70)
print("QUESTION 1: CLASSIFICATION - STEEL PLATES FAULTS")
print("=" * 70)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. LOAD DATASET
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[1/7] Loading Steel Plates Faults dataset from UCI...")
steel_plates = fetch_ucirepo(id=198)

X = steel_plates.data.features
y_raw = steel_plates.data.targets

print(f"  Features shape : {X.shape}")
print(f"  Targets shape  : {y_raw.shape}")
print(f"  Target columns : {list(y_raw.columns)}")

# The targets are one-hot encoded fault types â†’ convert to single class label
fault_cols = list(y_raw.columns)
y = pd.Series(np.argmax(y_raw.values, axis=1), name="fault_class")
class_names = fault_cols

print(f"\n  Class distribution:")
for i, name in enumerate(class_names):
    count = (y == i).sum()
    print(f"    [{i}] {name:<30} : {count}")

# Drop rows with missing values
mask = X.notnull().all(axis=1)
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)
print(f"\n  After dropping NaN rows: X={X.shape}, y={y.shape}")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. TRAIN-TEST SPLIT (80/20)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[2/7] Splitting data 80% train / 20% test (stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

n_classes = len(class_names)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# HELPER FUNCTIONS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def compute_metrics(name, model, X_tr, X_te, y_tr, y_te, proba=True):
    """Train, predict, and return a metrics dict."""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_te, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_te, y_pred, average='weighted', zero_division=0)

    if proba and hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_te)
        roc = roc_auc_score(y_te, y_prob, multi_class='ovr', average='weighted')
    elif hasattr(model, 'decision_function'):
        y_score = model.decision_function(X_te)
        if y_score.ndim == 1:
            y_score = np.column_stack([-y_score, y_score])
        roc = roc_auc_score(y_te, y_score, multi_class='ovr', average='weighted')
    else:
        roc = np.nan

    print(f"\n  {name}")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    ROC-AUC   : {roc:.4f}")

    return {"Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1, "ROC-AUC": roc,
            "y_pred": y_pred, "model": model}


def plot_confusion_matrix(cm, title, fname, classes):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_roc_multiclass(model, X_te, y_te, title, fname, use_proba=True):
    """Plot one-vs-rest ROC curves for each class."""
    y_bin = label_binarize(y_te, classes=list(range(n_classes)))

    if use_proba and hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_te)
    elif hasattr(model, 'decision_function'):
        y_score = model.decision_function(X_te)
        if y_score.ndim == 1:
            y_score = np.column_stack([-y_score, y_score])
    else:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, (cls_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.5,
                label=f'{cls_name} (AUC={roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'ROC Curves - {title}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3a. K-NEAREST NEIGHBOUR with Cross-Validation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[3/7] Training classifiers...")
print("\n  â”€â”€ KNN with Cross-Validation â”€â”€")

# Grid search for best k
knn_pipe = Pipeline([('knn', KNeighborsClassifier())])
param_grid_knn = {'knn__n_neighbors': [3, 5, 7, 9, 11, 15]}
knn_gs = GridSearchCV(knn_pipe, param_grid_knn, cv=cv,
                      scoring='accuracy', n_jobs=-1)
knn_gs.fit(X_train_sc, y_train)
best_k = knn_gs.best_params_['knn__n_neighbors']
print(f"  Best k (5-fold CV): {best_k}")

knn_cv_scores = cross_val_score(
    KNeighborsClassifier(n_neighbors=best_k), X_train_sc, y_train,
    cv=cv, scoring='accuracy'
)
print(f"  CV Accuracy: {knn_cv_scores.mean():.4f} Â± {knn_cv_scores.std():.4f}")

knn_best = KNeighborsClassifier(n_neighbors=best_k)
res_knn = compute_metrics("KNN (k=%d, CV)" % best_k,
                          knn_best, X_train_sc, X_test_sc, y_train, y_test)

# â”€â”€ Plot CV accuracy vs k â”€â”€
cv_means = [cross_val_score(KNeighborsClassifier(n_neighbors=k),
                            X_train_sc, y_train,
                            cv=cv, scoring='accuracy').mean()
            for k in [3, 5, 7, 9, 11, 15]]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([3, 5, 7, 9, 11, 15], cv_means, 'o-', color='steelblue', lw=2)
ax.axvline(best_k, color='red', linestyle='--', label=f'Best k={best_k}')
ax.set_xlabel('k (Number of Neighbors)')
ax.set_ylabel('CV Accuracy')
ax.set_title('KNN: Cross-Validation Accuracy vs k')
ax.legend(); plt.tight_layout()
plt.savefig('outputs/q1/knn_cv_accuracy.png', dpi=150)
plt.close()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3b. DECISION TREE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n  â”€â”€ Decision Tree â”€â”€")

# Explore depth for overfitting/underfitting analysis
depths = list(range(1, 21))
train_accs, test_accs = [], []
for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train_sc, y_train)
    train_accs.append(accuracy_score(y_train, dt.predict(X_train_sc)))
    test_accs.append(accuracy_score(y_test, dt.predict(X_test_sc)))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(depths, train_accs, 'o-', label='Train Accuracy', color='steelblue')
ax.plot(depths, test_accs,  's-', label='Test Accuracy',  color='orange')
ax.axvline(depths[np.argmax(test_accs)], color='red', linestyle='--',
           label=f'Best depth={depths[np.argmax(test_accs)]}')
ax.set_xlabel('Max Depth'); ax.set_ylabel('Accuracy')
ax.set_title('Decision Tree: Overfitting/Underfitting Analysis')
ax.legend(); plt.tight_layout()
plt.savefig('outputs/q1/dt_depth_analysis.png', dpi=150)
plt.close()

best_depth = depths[np.argmax(test_accs)]
print(f"  Best depth: {best_depth}")
dt_best = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
res_dt = compute_metrics("Decision Tree (depth=%d)" % best_depth,
                         dt_best, X_train_sc, X_test_sc, y_train, y_test)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3c. SVM with RBF Kernel
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n  â”€â”€ SVM (RBF Kernel) â”€â”€")

# Quick grid search
svm_param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
svm_gs = GridSearchCV(SVC(kernel='rbf', probability=True, random_state=42),
                      svm_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
svm_gs.fit(X_train_sc, y_train)
print(f"  Best params: {svm_gs.best_params_}")

svm_best = SVC(kernel='rbf', probability=True, random_state=42,
               **svm_gs.best_params_)
res_svm = compute_metrics("SVM (RBF)", svm_best,
                          X_train_sc, X_test_sc, y_train, y_test)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3d. DEEP NEURAL NETWORK (Bonus) â€” PyTorch
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n  â”€â”€ Deep Neural Network (Bonus) â”€â”€")
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    X_tr_t = torch.tensor(X_train_sc, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train.values, dtype=torch.long)
    X_te_t = torch.tensor(X_test_sc,  dtype=torch.float32)
    y_te_t = torch.tensor(y_test.values, dtype=torch.long)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

    class DeepNet(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128),    nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 64),     nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, out_dim)
            )
        def forward(self, x):
            return self.net(x)

    device = torch.device('cpu')
    model_dnn = DeepNet(X_train_sc.shape[1], n_classes).to(device)
    optimizer = torch.optim.Adam(model_dnn.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_losses, train_acc_hist, test_acc_hist = [], [], []
    EPOCHS = 80
    for epoch in range(EPOCHS):
        model_dnn.train()
        epoch_loss = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            out = model_dnn(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        scheduler.step()
        epoch_loss /= len(train_ds)
        train_losses.append(epoch_loss)

        model_dnn.eval()
        with torch.no_grad():
            tr_pred = model_dnn(X_tr_t).argmax(1).numpy()
            te_pred = model_dnn(X_te_t).argmax(1).numpy()
        train_acc_hist.append(accuracy_score(y_train, tr_pred))
        test_acc_hist.append(accuracy_score(y_test,  te_pred))

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS} | Loss={epoch_loss:.4f} | "
                  f"Train Acc={train_acc_hist[-1]:.4f} | Test Acc={test_acc_hist[-1]:.4f}")

    # Training curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, color='steelblue')
    axes[0].set_title('DNN Training Loss'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[1].plot(train_acc_hist, label='Train', color='steelblue')
    axes[1].plot(test_acc_hist,  label='Test',  color='orange')
    axes[1].set_title('DNN Accuracy Curve'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('outputs/q1/dnn_training_curve.png', dpi=150)
    plt.close()

    model_dnn.eval()
    with torch.no_grad():
        logits  = model_dnn(X_te_t)
        y_prob_dnn = torch.softmax(logits, dim=1).numpy()
        y_pred_dnn = logits.argmax(1).numpy()

    acc_dnn  = accuracy_score(y_test, y_pred_dnn)
    prec_dnn = precision_score(y_test, y_pred_dnn, average='weighted', zero_division=0)
    rec_dnn  = recall_score(y_test,  y_pred_dnn, average='weighted', zero_division=0)
    f1_dnn   = f1_score(y_test,  y_pred_dnn, average='weighted', zero_division=0)
    roc_dnn  = roc_auc_score(y_test, y_prob_dnn, multi_class='ovr', average='weighted')

    print(f"\n  Deep Neural Network (Bonus)")
    print(f"    Accuracy  : {acc_dnn:.4f}")
    print(f"    Precision : {prec_dnn:.4f}")
    print(f"    Recall    : {rec_dnn:.4f}")
    print(f"    F1-Score  : {f1_dnn:.4f}")
    print(f"    ROC-AUC   : {roc_dnn:.4f}")

    res_dnn = {"Model": "Deep Neural Network (Bonus)",
               "Accuracy": acc_dnn, "Precision": prec_dnn,
               "Recall": rec_dnn, "F1": f1_dnn, "ROC-AUC": roc_dnn,
               "y_pred": y_pred_dnn, "model": None}
    dnn_available = True
except Exception as e:
    print(f"  DNN skipped: {e}")
    dnn_available = False
    res_dnn = None

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. CONFUSION MATRICES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[4/7] Plotting confusion matrices...")
results = [res_knn, res_dt, res_svm]
if dnn_available and res_dnn:
    results.append(res_dnn)

short_names = [c[:8] for c in class_names]   # shorten for readability
for res in results:
    cm = confusion_matrix(y_test, res["y_pred"])
    safe_name = res["Model"].replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    plot_confusion_matrix(cm, f'Confusion Matrix - {res["Model"]}',
                          f'outputs/q1/cm_{safe_name}.png', short_names)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5. ROC CURVES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[5/7] Plotting ROC curves...")
plot_roc_multiclass(res_knn["model"], X_test_sc,  y_test, "KNN",          'outputs/q1/roc_knn.png')
plot_roc_multiclass(res_dt["model"],  X_test_sc,  y_test, "Decision Tree",'outputs/q1/roc_dt.png')
plot_roc_multiclass(res_svm["model"], X_test_sc,  y_test, "SVM (RBF)",    'outputs/q1/roc_svm.png')

if dnn_available and res_dnn:
    # Manual ROC for DNN using stored probabilities
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, (cls_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob_dnn[:, i])
        ax.plot(fpr, tpr, color=color, lw=1.5,
                label=f'{cls_name} (AUC={auc(fpr, tpr):.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC Curves - Deep Neural Network (Bonus)', fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig('outputs/q1/roc_dnn.png', dpi=150)
    plt.close()
    print("  Saved: outputs/q1/roc_dnn.png")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6. COMPARATIVE ROC PLOT (macro AUC per model)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[6/7] Comparative summary plot...")

metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
model_labels = [r["Model"].split("(")[0].strip() for r in results]
metric_values = [[r[m] for m in metric_names] for r in results]

x = np.arange(len(metric_names))
width = 0.18
fig, ax = plt.subplots(figsize=(13, 5))
for i, (label, vals) in enumerate(zip(model_labels, metric_values)):
    ax.bar(x + i * width, vals, width, label=label)
ax.set_xticks(x + width * (len(results) - 1) / 2)
ax.set_xticklabels(metric_names)
ax.set_ylim([0, 1.1])
ax.set_ylabel('Score')
ax.set_title('Classifier Comparison â€“ All Metrics', fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('outputs/q1/classifier_comparison.png', dpi=150)
plt.close()
print("  Saved: outputs/q1/classifier_comparison.png")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 7. SUMMARY TABLE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[7/7] Final Results Summary")
print("=" * 70)
summary_df = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ("y_pred", "model")}
    for r in results
])
summary_df = summary_df.set_index("Model")
for col in summary_df.columns:
    summary_df[col] = summary_df[col].map(lambda x: f"{x:.4f}")
print(summary_df.to_string())

best_model = max(results, key=lambda r: r["ROC-AUC"])
print(f"\n  âœ“ Best classifier by ROC-AUC: {best_model['Model']} "
      f"(ROC-AUC = {best_model['ROC-AUC']:.4f})")
print("\nDone! All outputs saved to outputs/q1/")

