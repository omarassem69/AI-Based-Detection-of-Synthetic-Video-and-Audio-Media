import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve

# =========================
# CONFIG
# =========================
IN_DIR = r"results/audio_new"
OUT_DIR = r"results/audio_new"  # save outputs next to your audio results
POS_LABEL = 1  # class 1 = spoof/fake (positive)

y_test_path = os.path.join(IN_DIR, "y_test.npy")
y_prob_path = os.path.join(IN_DIR, "y_prob.npy")

assert os.path.exists(y_test_path), f"Missing: {y_test_path}"
assert os.path.exists(y_prob_path), f"Missing: {y_prob_path}"

y_true = np.load(y_test_path).astype(int)
y_score = np.load(y_prob_path).reshape(-1)

# =========================
# ROC-AUC
# =========================
auc = roc_auc_score(y_true, y_score)

# =========================
# ROC + EER
# EER is where FPR == FNR (i.e., FPR == 1 - TPR)
# We'll compute it from ROC curve by finding the point minimizing |FPR - FNR|
# =========================
fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=POS_LABEL)
fnr = 1 - tpr

eer_idx = np.argmin(np.abs(fpr - fnr))
eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
eer_threshold = thresholds[eer_idx]

# =========================
# Save ROC curve plot
# =========================
plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
plt.scatter([fpr[eer_idx]], [tpr[eer_idx]], label=f"EER={eer:.4f} @ thr={eer_threshold:.4f}")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Audio ROC Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
roc_path = os.path.join(OUT_DIR, "roc_curve_audio.png")
plt.savefig(roc_path, dpi=300)
plt.close()

# =========================
# Save text summary
# =========================
out_txt = os.path.join(OUT_DIR, "audio_eer_auc.txt")
with open(out_txt, "w", encoding="utf-8") as f:
    f.write("Audio Evaluation Metrics\n")
    f.write("========================\n")
    f.write(f"ROC-AUC: {auc:.6f}\n")
    f.write(f"EER:     {eer:.6f}\n")
    f.write(f"EER thr: {eer_threshold:.6f}\n")

print("✅ Audio ROC-AUC:", auc)
print("✅ Audio EER    :", eer)
print("✅ EER threshold:", eer_threshold)
print("\nSaved:")
print(" -", out_txt)
print(" -", roc_path)
