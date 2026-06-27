import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

IN_DIR = r"results/video_tuned_safe"
y_path = os.path.join(IN_DIR, "y_test.npy")
p_path = os.path.join(IN_DIR, "p_test.npy")

assert os.path.exists(y_path), f"Missing: {y_path}"
assert os.path.exists(p_path), f"Missing: {p_path}"

y_true = np.load(y_path).astype(int)
y_score = np.load(p_path).reshape(-1)  # prob(fake)

auc = roc_auc_score(y_true, y_score)

fpr, tpr, thr = roc_curve(y_true, y_score, pos_label=1)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Video ROC Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()

roc_path = os.path.join(IN_DIR, "roc_curve_video.png")
plt.savefig(roc_path, dpi=300)
plt.close()

txt_path = os.path.join(IN_DIR, "video_auc.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("Video Evaluation Metrics\n")
    f.write("========================\n")
    f.write(f"ROC-AUC: {auc:.6f}\n")

print("✅ Video ROC-AUC:", auc)
print("Saved:")
print(" -", txt_path)
print(" -", roc_path)
