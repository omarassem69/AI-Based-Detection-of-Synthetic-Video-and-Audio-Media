import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
X_PATH = r"data/output/audio/X_cvoicefake_en.npy"
Y_PATH = r"data/output/audio/y_cvoicefake_en.npy"
MODEL_PATH = r"models/audio_model_final.keras"  # change if your audio model name differs
OUT_DIR = r"results/audio_new"
SEED = 42
THRESH = 0.5  # spoof/fake if prob >= 0.5 (assumes label 1 = spoof/fake)

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)


def save_confusion_matrix(cm, labels, out_path, title):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


print("Loading audio features...")
X = np.load(X_PATH)
y = np.load(Y_PATH).astype(int)

print("Original distribution:", dict(zip(*np.unique(y, return_counts=True))))

# keep same split pattern as video for consistency
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
)

print("Test distribution:", dict(zip(*np.unique(y_test, return_counts=True))))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Audio model not found at: {MODEL_PATH}")

print("Loading model:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
print("Model output shape:", model.output_shape)

print("Predicting on test set...")
# memory-friendly prediction
y_prob_raw = model.predict(X_test, verbose=0, batch_size=32)

# support sigmoid (N,1) and softmax (N,2)
if y_prob_raw.ndim == 2 and y_prob_raw.shape[1] == 1:
    y_prob = y_prob_raw.reshape(-1)          # prob(class=1)
elif y_prob_raw.ndim == 2 and y_prob_raw.shape[1] == 2:
    # common: index 1 is spoof/fake
    y_prob = y_prob_raw[:, 1].reshape(-1)
else:
    y_prob = y_prob_raw.reshape(-1)

y_pred = (y_prob >= THRESH).astype(int)

print("Pred distribution:", dict(zip(*np.unique(y_pred, return_counts=True))))
print("Prob stats: min=%.4f max=%.4f mean=%.4f" % (y_prob.min(), y_prob.max(), y_prob.mean()))

# metrics (positive class = 1)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\n=== Overall Metrics (class 1 is positive) ===")
print("Accuracy   :", acc)
print("Precision  :", prec)
print("Recall     :", rec)
print("F1         :", f1)
print("Macro-F1   :", macro_f1)
print("Weighted-F1:", weighted_f1)

# label names (adjust if your dataset uses Bonafide/Spoof)
report = classification_report(
    y_test, y_pred,
    target_names=["Class 0", "Class 1"],
    digits=4,
    zero_division=0
)
print("\n=== Classification Report ===")
print(report)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print("\n=== Confusion Matrix (rows=true, cols=pred) ===")
print(cm)

# save
with open(os.path.join(OUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write("THRESH=%.2f\n" % THRESH)
    f.write("Accuracy=%.6f\n" % acc)
    f.write("MacroF1=%.6f\n" % macro_f1)
    f.write("WeightedF1=%.6f\n\n" % weighted_f1)
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))

np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)
np.save(os.path.join(OUT_DIR, "y_prob.npy"), y_prob)
np.save(os.path.join(OUT_DIR, "y_pred.npy"), y_pred)

save_confusion_matrix(
    cm,
    labels=["Class 0", "Class 1"],
    out_path=os.path.join(OUT_DIR, "confusion_matrix.png"),
    title="Audio Confusion Matrix (Test)"
)

print("\nSaved to:", OUT_DIR)
print("Done ✅")
