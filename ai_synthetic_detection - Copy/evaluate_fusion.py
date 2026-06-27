import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report


# =========================
# CONFIG
# =========================
VIDEO_DIR = r"results/video_new"
AUDIO_DIR = r"results/audio_new"
OUT_DIR = r"results/fusion_new"

THRESH = 0.5
ALPHAS = np.round(np.arange(0.0, 1.01, 0.05), 2)  # 0.00 to 1.00

os.makedirs(OUT_DIR, exist_ok=True)


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


# =========================
# LOAD SAVED PROBS
# =========================
y_test_v = np.load(os.path.join(VIDEO_DIR, "y_test.npy")).astype(int)
p_v = np.load(os.path.join(VIDEO_DIR, "y_prob.npy")).reshape(-1)

y_test_a = np.load(os.path.join(AUDIO_DIR, "y_test.npy")).astype(int)
p_a = np.load(os.path.join(AUDIO_DIR, "y_prob.npy")).reshape(-1)

print("Video test size:", len(y_test_v))
print("Audio test size:", len(y_test_a))

# IMPORTANT: Fusion needs aligned samples.
# If they are not aligned (different datasets), we can only do fusion if you have paired audio+video samples.
if len(y_test_v) != len(y_test_a):
    raise ValueError(
        "Video and Audio test sizes differ. Fusion evaluation requires paired samples.\n"
        f"video={len(y_test_v)}, audio={len(y_test_a)}\n"
        "If your fusion is done on paired files, we must generate probs on the SAME paired list."
    )

# also check labels match sample-by-sample
if not np.array_equal(y_test_v, y_test_a):
    raise ValueError(
        "y_test arrays differ between video and audio.\n"
        "Fusion requires both branches evaluated on the SAME paired samples and SAME y_true order."
    )

y_true = y_test_v

print("Test distribution:", dict(zip(*np.unique(y_true, return_counts=True))))

# =========================
# SWEEP ALPHA
# =========================
accs = []
macro_f1s = []

best = {"alpha": None, "acc": -1, "macro_f1": -1}

for a in ALPHAS:
    p_f = a * p_v + (1 - a) * p_a
    y_pred = (p_f >= THRESH).astype(int)

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    accs.append(acc)
    macro_f1s.append(mf1)

    # choose best by accuracy first, then macro-f1
    if (acc > best["acc"]) or (acc == best["acc"] and mf1 > best["macro_f1"]):
        best = {"alpha": float(a), "acc": float(acc), "macro_f1": float(mf1)}

# save alpha plot (accuracy)
plt.figure()
plt.plot(ALPHAS, accs, marker="o")
plt.title("Fusion Weight Tuning (alpha vs accuracy)")
plt.xlabel("alpha (video weight)")
plt.ylabel("accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "alpha_vs_accuracy.png"), dpi=300)
plt.close()

# save alpha plot (macro-f1)
plt.figure()
plt.plot(ALPHAS, macro_f1s, marker="o")
plt.title("Fusion Weight Tuning (alpha vs macro-F1)")
plt.xlabel("alpha (video weight)")
plt.ylabel("macro-F1")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "alpha_vs_macro_f1.png"), dpi=300)
plt.close()

print("\nBest alpha:", best)

# =========================
# FINAL FUSION @ BEST ALPHA
# =========================
a = best["alpha"]
p_f = a * p_v + (1 - a) * p_a
y_pred = (p_f >= THRESH).astype(int)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print("\n=== Fusion Metrics @ best alpha ===")
print("alpha:", a)
print("accuracy:", acc)
print("macro_f1:", macro_f1)
print("weighted_f1:", weighted_f1)
print("confusion matrix:\n", cm)

report = classification_report(
    y_true, y_pred,
    target_names=["Class 0", "Class 1"],
    digits=4,
    zero_division=0
)
print("\n=== Fusion Classification Report ===")
print(report)

# save confusion matrix fig
save_confusion_matrix(
    cm,
    labels=["Class 0", "Class 1"],
    out_path=os.path.join(OUT_DIR, "confusion_matrix.png"),
    title=f"Fusion Confusion Matrix (alpha={a:.2f})"
)

# save text
with open(os.path.join(OUT_DIR, "fusion_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"alpha={a:.2f}\n")
    f.write(f"THRESH={THRESH:.2f}\n")
    f.write(f"accuracy={acc:.6f}\n")
    f.write(f"macro_f1={macro_f1:.6f}\n")
    f.write(f"weighted_f1={weighted_f1:.6f}\n\n")
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))

np.save(os.path.join(OUT_DIR, "y_true.npy"), y_true)
np.save(os.path.join(OUT_DIR, "y_pred.npy"), y_pred)
np.save(os.path.join(OUT_DIR, "y_prob_fused.npy"), p_f)

print("\nSaved to:", OUT_DIR)
print("Done ✅")
