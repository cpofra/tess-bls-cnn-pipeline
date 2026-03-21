# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 12:00:16 2026

@author: 14358
"""
# -*- coding: utf-8 -*-
"""
TESS Phase 3: Train CNN on Phase 2 transit-centered windows

Input:
  - tess_phase2_windows.npz

Output:
  - best_phase3_model.keras
  - final_phase3_model.keras
  - phase3_thresholds.json
  - phase3_inference_config.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# ----------------------------
# Config
# ----------------------------
NPZ_PATH = "tess_phase2_windows.npz"
RANDOM_STATE = 42

EPOCHS = 100
BATCH_SIZE = 128
LR = 1e-3

USE_FOCAL_LOSS = False   # set True if you want fewer false positives in sweeps
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25       # standard starting point

DEFAULT_THRESH = 0.86
LIKELY_THRESH = 0.88
HIGH_CONF_THRESH = 0.895

THRESH_JSON = "phase3_thresholds.json"
INFER_JSON = "phase3_inference_config.json"


# ----------------------------
# Losses
# ----------------------------
def binary_focal_loss(gamma=2.0, alpha=0.25):
    """
    Binary focal loss for logits=probabilities (sigmoid output).
    """
    gamma = float(gamma)
    alpha = float(alpha)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # p_t
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)

        # alpha factor
        a_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)

        fl = -a_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(fl)

    return loss


# ----------------------------
# Threshold utilities
# ----------------------------
def threshold_report(y_true, y_score, thr):
    y_pred = (y_score >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return {
        "threshold": float(thr),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def find_thresholds(y_true, y_score, want_precision=0.90):
    # Best F1 from PR curve
    prec, rec, thr_pr = precision_recall_curve(y_true, y_score)
    # prec/rec arrays are length len(thr)+1
    f1 = 2 * (prec * rec) / (prec + rec + 1e-12)
    best_f1_idx = int(np.nanargmax(f1))
    # Map index to threshold: last element corresponds to no threshold
    if best_f1_idx == len(thr_pr):
        best_f1_thr = 0.5
    else:
        best_f1_thr = float(thr_pr[best_f1_idx])

    # Best Youden J from ROC curve
    fpr, tpr, thr_roc = roc_curve(y_true, y_score)
    J = tpr - fpr
    best_j_idx = int(np.nanargmax(J))
    best_j_thr = float(thr_roc[best_j_idx])

    # Threshold achieving >= want_precision (highest recall among those)
    # We can scan unique scores as candidate thresholds
    candidates = np.unique(y_score)
    candidates = np.sort(candidates)

    best_p_thr = None
    best_p_rec = -1.0

    for thr in candidates:
        rep = threshold_report(y_true, y_score, thr)
        if rep["precision"] >= want_precision and rep["recall"] > best_p_rec:
            best_p_rec = rep["recall"]
            best_p_thr = float(thr)

    return {
        "best_f1": threshold_report(y_true, y_score, best_f1_thr),
        "best_youden_j": threshold_report(y_true, y_score, best_j_thr),
        f"precision_ge_{want_precision:.2f}": threshold_report(
            y_true, y_score, best_p_thr if best_p_thr is not None else 0.95
        ),
        "default_0.5": threshold_report(y_true, y_score, 0.5),
    }


def label_tiers(y_score, t_candidate=0.50, t_likely=0.85, t_high=0.90):
    labels = np.full(len(y_score), "reject", dtype=object)
    labels[y_score >= t_candidate] = "candidate"
    labels[y_score >= t_likely] = "likely_planet"
    labels[y_score >= t_high] = "high_confidence_planet"
    return labels


# ----------------------------
# Load
# ----------------------------
d = np.load(NPZ_PATH, allow_pickle=True)
X = d["X"].astype(np.float32)    # (N, 301, 1)
y = d["y"].astype(np.int64)      # (N,)

print("Loaded:", NPZ_PATH)
print("X:", X.shape, X.dtype)
print("y counts:", {0: int(np.sum(y == 0)), 1: int(np.sum(y == 1))})

assert X.ndim == 3 and X.shape[-1] == 1
assert set(np.unique(y)).issubset({0, 1})


# ----------------------------
# Train/val/test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.20, random_state=RANDOM_STATE, stratify=y_train
)

print("\nSplit:")
print("  train:", X_train.shape, "pos:", int(np.sum(y_train == 1)), "neg:", int(np.sum(y_train == 0)))
print("  val:  ", X_val.shape,   "pos:", int(np.sum(y_val == 1)),   "neg:", int(np.sum(y_val == 0)))
print("  test: ", X_test.shape,  "pos:", int(np.sum(y_test == 1)),  "neg:", int(np.sum(y_test == 0)))

# Class weights
pos = np.sum(y_train == 1)
neg = np.sum(y_train == 0)
class_weight = {
    0: (len(y_train) / (2.0 * neg)),
    1: (len(y_train) / (2.0 * pos)),
}
print("\nclass_weight:", class_weight)


# ----------------------------
# Model
# ----------------------------
def build_model(input_shape=(301, 1)) -> keras.Model:
    inp = layers.Input(shape=input_shape)

    x = layers.Conv1D(32, 9, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Conv1D(64, 7, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)

    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="Phase3_TESS_Window_CNN")


model = build_model(input_shape=X_train.shape[1:])

loss_fn = binary_focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA) if USE_FOCAL_LOSS else "binary_crossentropy"

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss=loss_fn,
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
    ],
)

model.summary()


# ----------------------------
# Train
# ----------------------------
callbacks = [
    EarlyStopping(monitor="val_auc", patience=12, mode="max", restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=6, mode="max", min_lr=1e-6, verbose=1),
    ModelCheckpoint("best_phase3_model.keras", monitor="val_auc", mode="max", save_best_only=True, verbose=1),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)


# ----------------------------
# Evaluate
# ----------------------------
print("\nEvaluating on test set...")
test = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss={test[0]:.4f}  Acc={test[1]:.4f}  Prec={test[2]:.4f}  Recall={test[3]:.4f}  AUC={test[4]:.4f}")

y_proba = model.predict(X_test, verbose=0).ravel()
y_pred = (y_proba >= DEFAULT_THRESH).astype(int)

print("\nClassification report @ 0.5:")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
print("ROC AUC:", roc_auc)

# ----------------------------
# Threshold recommendations (saved)
# ----------------------------
print("\n==============================")
print("Threshold recommendations")
print("==============================")
thr_info = find_thresholds(y_test, y_proba, want_precision=0.90)

for k, v in thr_info.items():
    print(f"\n▶ {k}")
    print(v)

with open(THRESH_JSON, "w") as f:
    json.dump(thr_info, f, indent=2)
print(f"\n✓ Saved thresholds: {THRESH_JSON}")

# ----------------------------
# Use FIXED sweep thresholds (these drive Phase 5 + reporting)
# ----------------------------
CAND_THRESH = float(DEFAULT_THRESH)        # <-- 0.86
LIKELY_T    = float(LIKELY_THRESH)         # <-- 0.88
HIGH_T      = float(HIGH_CONF_THRESH)      # <-- 0.895

# Label tiers using your desired defaults (these are for sweep labeling)
labels = label_tiers(y_proba, CAND_THRESH, LIKELY_T, HIGH_T)
tier_counts = {lab: int(np.sum(labels == lab)) for lab in np.unique(labels)}
print("\nTier counts on TEST (using fixed sweep thresholds):", tier_counts)

# Save inference config (so Phase 5 reads the same thresholds)
infer_cfg = {
    "window_len": int(X.shape[1]),
    "candidate_threshold": CAND_THRESH,
    "likely_threshold": LIKELY_T,
    "high_confidence_threshold": HIGH_T,

    # optional metadata
    "use_focal_loss": bool(USE_FOCAL_LOSS),
    "focal_gamma": float(FOCAL_GAMMA),
    "focal_alpha": float(FOCAL_ALPHA),
}
with open(INFER_JSON, "w") as f:
    json.dump(infer_cfg, f, indent=2)
print(f"✓ Saved inference config: {INFER_JSON}")


# ----------------------------
# Plots
# ----------------------------
plt.figure()
plt.plot(history.history["auc"])
plt.plot(history.history["val_auc"])
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.legend(["train", "val"])
plt.title("AUC history")
plt.grid(True, alpha=0.3)
plt.show()

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (AUC={roc_auc:.3f})")
plt.grid(True, alpha=0.3)
plt.show()


# ----------------------------
# Save final model
# ----------------------------
model.save("final_phase3_model.keras")
print("\n✓ Saved: best_phase3_model.keras")
print("✓ Saved: final_phase3_model.keras")
