import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)

# =========================
# CONFIG (matches your training/eval)
# =========================
X_PATH = r"data/output/video/X_ffpp.npy"
Y_PATH = r"data/output/video/y_ffpp.npy"
MODEL_PATH = r"models/video_model_balanced.keras"
OUT_DIR = r"results/video_tuned_safe"

IMG_BATCH = 16
SEED = 42
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
SELECT_BY = "macro_f1"  # best for imbalance

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


def metrics(y_true, p_fake, thr):
    y_pred = (p_fake >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Fake=1 positive
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    return {
        "thr": float(thr),
        "acc": float(acc),
        "bacc": float(bacc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "prec_fake": float(prec),
        "rec_fake": float(rec),
        "f1_fake": float(f1),
        "y_pred": y_pred,
    }


print("Loading data...")
X = np.load(X_PATH)
y = np.load(Y_PATH).astype(int)

print("Original distribution:", dict(zip(*np.unique(y, return_counts=True))))

# same split pattern you used before:
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
)

print("Val distribution:", dict(zip(*np.unique(y_val, return_counts=True))))
print("Test distribution:", dict(zip(*np.unique(y_test, return_counts=True))))

print("Loading model:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
print("Model output shape:", model.output_shape)

print("Predicting probabilities...")
p_val = model.predict(X_val, verbose=0, batch_size=IMG_BATCH).reshape(-1)
p_test = model.predict(X_test, verbose=0, batch_size=IMG_BATCH).reshape(-1)

print("p_val stats:  min=%.4f max=%.4f mean=%.4f" % (p_val.min(), p_val.max(), p_val.mean()))
print("p_test stats: min=%.4f max=%.4f mean=%.4f" % (p_test.min(), p_test.max(), p_test.mean()))

# -------------------------
# Tune threshold on VAL
# -------------------------
rows = []
best = None

for thr in THRESHOLDS:
    m = metrics(y_val, p_val, thr)
    rows.append(m)

    if best is None:
        best = m
    else:
        if (m[SELECT_BY] > best[SELECT_BY]) or (
            m[SELECT_BY] == best[SELECT_BY] and (m["macro_f1"], m["bacc"]) > (best["macro_f1"], best["bacc"])
        ):
            best = m

print("\n=== VAL Threshold sweep summary ===")
print("thr   acc     bacc    macroF1  wF1     prec(F) rec(F)  F1(F)")
for m in rows:
    print(
        f"{m['thr']:<4.2f}  "
        f"{m['acc']:<7.4f} "
        f"{m['bacc']:<7.4f} "
        f"{m['macro_f1']:<7.4f} "
        f"{m['weighted_f1']:<7.4f} "
        f"{m['prec_fake']:<7.4f} "
        f"{m['rec_fake']:<7.4f} "
        f"{m['f1_fake']:<7.4f}"
    )

best_thr = best["thr"]
print("\n✅ BEST THRESHOLD chosen on VAL (by %s): %.2f" % (SELECT_BY, best_thr))

# -------------------------
# Evaluate chosen threshold on TEST
# -------------------------
m_test = metrics(y_test, p_test, best_thr)
y_pred_test = m_test["y_pred"]

cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
report = classification_report(
    y_test, y_pred_test,
    target_names=["Real (0)", "Fake (1)"],
    digits=4,
    zero_division=0
)

print("\n=== TEST Metrics @ chosen threshold ===")
print("Threshold:", best_thr)
print("Accuracy:", m_test["acc"])
print("Balanced Acc:", m_test["bacc"])
print("Macro-F1:", m_test["macro_f1"])
print("Weighted-F1:", m_test["weighted_f1"])
print("Fake Precision/Recall/F1:", m_test["prec_fake"], m_test["rec_fake"], m_test["f1_fake"])
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# -------------------------
# Save artifacts
# -------------------------
with open(os.path.join(OUT_DIR, "video_threshold_tuned_SAFE.txt"), "w", encoding="utf-8") as f:
    f.write(f"SELECT_BY={SELECT_BY}\n")
    f.write(f"BEST_THRESHOLD_VAL={best_thr:.2f}\n\n")
    f.write("=== TEST METRICS @ chosen threshold ===\n")
    f.write(f"Accuracy={m_test['acc']:.6f}\n")
    f.write(f"BalancedAcc={m_test['bacc']:.6f}\n")
    f.write(f"MacroF1={m_test['macro_f1']:.6f}\n")
    f.write(f"WeightedF1={m_test['weighted_f1']:.6f}\n")
    f.write(f"FakePrecision={m_test['prec_fake']:.6f}\n")
    f.write(f"FakeRecall={m_test['rec_fake']:.6f}\n")
    f.write(f"FakeF1={m_test['f1_fake']:.6f}\n\n")
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))

np.save(os.path.join(OUT_DIR, "best_threshold_val.npy"), np.array([best_thr], dtype=np.float32))
np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)
np.save(os.path.join(OUT_DIR, "y_pred_test.npy"), y_pred_test)
np.save(os.path.join(OUT_DIR, "p_test.npy"), p_test)

save_confusion_matrix(
    cm,
    labels=["Real", "Fake"],
    out_path=os.path.join(OUT_DIR, "confusion_matrix_TEST.png"),
    title=f"Video Confusion Matrix (TEST, thr={best_thr:.2f})"
)

# plot VAL curves
ths = [m["thr"] for m in rows]
accs = [m["acc"] for m in rows]
baccs = [m["bacc"] for m in rows]
mf1s = [m["macro_f1"] for m in rows]

plt.figure()
plt.plot(ths, accs, marker="o", label="VAL Accuracy")
plt.plot(ths, baccs, marker="o", label="VAL Balanced Acc")
plt.plot(ths, mf1s, marker="o", label="VAL Macro-F1")
plt.xlabel("Threshold (Fake if p>=thr)")
plt.ylabel("Score")
plt.title("Video Threshold Tuning on Validation Set")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "threshold_tuning_VAL.png"), dpi=300)
plt.close()

print("\nSaved to:", OUT_DIR)
print("Done ✅")
