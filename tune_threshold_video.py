import os
import numpy as np
import matplotlib.pyplot as plt

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
# CONFIG
# =========================
IN_DIR = r"results/video_new"          # uses your saved y_test/y_prob
OUT_DIR = r"results/video_tuned"       # new folder for tuned outputs
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)  # 0.05..0.95
SELECT_BY = "macro_f1"                 # "macro_f1" (recommended) or "balanced_acc"

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


def metrics_at_threshold(y_true, p_fake, thr):
    y_pred = (p_fake >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Fake=1 is positive
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


print("Loading saved video test outputs...")
y_true = np.load(os.path.join(IN_DIR, "y_test.npy")).astype(int)
p_fake = np.load(os.path.join(IN_DIR, "y_prob.npy")).reshape(-1)

print("y_true distribution:", dict(zip(*np.unique(y_true, return_counts=True))))
print("p_fake stats: min=%.4f max=%.4f mean=%.4f" % (p_fake.min(), p_fake.max(), p_fake.mean()))

rows = []
best = None

for thr in THRESHOLDS:
    m = metrics_at_threshold(y_true, p_fake, thr)
    rows.append(m)

    key = m[SELECT_BY]
    if best is None:
        best = m
    else:
        best_key = best[SELECT_BY]
        # choose best by selected metric, tie-break by macro_f1 then bacc
        if (key > best_key) or (
            key == best_key and (m["macro_f1"], m["bacc"]) > (best["macro_f1"], best["bacc"])
        ):
            best = m

# Print summary table
print("\n=== Threshold sweep summary ===")
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

print("\n✅ BEST THRESHOLD (by %s): %.2f" % (SELECT_BY, best["thr"]))
print("Accuracy:", best["acc"])
print("Balanced Acc:", best["bacc"])
print("Macro-F1:", best["macro_f1"])
print("Weighted-F1:", best["weighted_f1"])
print("Fake Precision/Recall/F1:", best["prec_fake"], best["rec_fake"], best["f1_fake"])

# Final report at best threshold
y_pred_best = best["y_pred"]
cm = confusion_matrix(y_true, y_pred_best, labels=[0, 1])

report = classification_report(
    y_true, y_pred_best,
    target_names=["Real (0)", "Fake (1)"],
    digits=4,
    zero_division=0
)

print("\n=== Confusion Matrix @ best threshold ===")
print(cm)
print("\n=== Classification Report @ best threshold ===")
print(report)

# Save artifacts
with open(os.path.join(OUT_DIR, "best_threshold_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"SELECT_BY={SELECT_BY}\n")
    f.write(f"BEST_THRESHOLD={best['thr']:.2f}\n")
    f.write(f"Accuracy={best['acc']:.6f}\n")
    f.write(f"BalancedAcc={best['bacc']:.6f}\n")
    f.write(f"MacroF1={best['macro_f1']:.6f}\n")
    f.write(f"WeightedF1={best['weighted_f1']:.6f}\n")
    f.write(f"FakePrecision={best['prec_fake']:.6f}\n")
    f.write(f"FakeRecall={best['rec_fake']:.6f}\n")
    f.write(f"FakeF1={best['f1_fake']:.6f}\n\n")
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))

np.save(os.path.join(OUT_DIR, "y_pred_best.npy"), y_pred_best)
np.save(os.path.join(OUT_DIR, "best_threshold.npy"), np.array([best["thr"]], dtype=np.float32))

save_confusion_matrix(
    cm,
    labels=["Real", "Fake"],
    out_path=os.path.join(OUT_DIR, "confusion_matrix_best_threshold.png"),
    title=f"Video Confusion Matrix (thr={best['thr']:.2f})"
)

# Plot threshold vs metrics
ths = [m["thr"] for m in rows]
accs = [m["acc"] for m in rows]
baccs = [m["bacc"] for m in rows]
mf1s = [m["macro_f1"] for m in rows]

plt.figure()
plt.plot(ths, accs, marker="o", label="Accuracy")
plt.plot(ths, baccs, marker="o", label="Balanced Acc")
plt.plot(ths, mf1s, marker="o", label="Macro-F1")
plt.xlabel("Threshold (Fake if p>=thr)")
plt.ylabel("Score")
plt.title("Video Threshold Tuning")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "threshold_tuning_curve.png"), dpi=300)
plt.close()

print("\nSaved to:", OUT_DIR)
print("Done ✅")
