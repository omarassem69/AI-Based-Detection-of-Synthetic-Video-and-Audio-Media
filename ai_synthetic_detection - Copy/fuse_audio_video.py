import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# =========================
# PATHS (CONFIRMED)
# =========================
VIDEO_MODEL_PATH = r"C:\Users\omara\ai_synthetic_detection\data\output\result video\best_video_model.h5"
AUDIO_MODEL_PATH = r"C:\Users\omara\ai_synthetic_detection\data\output\result audio\audio_model_final.keras"

X_VIDEO_PATH = r"C:\Users\omara\ai_synthetic_detection\data\output\video\X_ffpp.npy"
Y_VIDEO_PATH = r"C:\Users\omara\ai_synthetic_detection\data\output\video\y_ffpp.npy"

X_AUDIO_PATH = r"C:\Users\omara\ai_synthetic_detection\data\output\audio\X_cvoicefake_en.npy"
Y_AUDIO_PATH = r"C:\Users\omara\ai_synthetic_detection\data\output\audio\y_cvoicefake_en.npy"

OUTPUT_DIR = r"C:\Users\omara\ai_synthetic_detection\data\output\fused_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD MODELS
# =========================
print("Loading models...")
video_model = tf.keras.models.load_model(VIDEO_MODEL_PATH)
audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)

# =========================
# LOAD DATA
# =========================
print("Loading data...")
Xv = np.load(X_VIDEO_PATH)
yv = np.load(Y_VIDEO_PATH)

Xa = np.load(X_AUDIO_PATH)
ya = np.load(Y_AUDIO_PATH)

# =========================
# ALIGN DATA
# =========================
N = min(len(yv), len(ya))
Xv, yv = Xv[:N], yv[:N]
Xa, ya = Xa[:N], ya[:N]

Xa = Xa[..., np.newaxis]  # (N, 40, 400, 1)

print(f"Aligned samples: {N}")

# =========================
# PREDICTIONS
# =========================
print("Running predictions...")
pv = video_model.predict(Xv, batch_size=32).ravel()
pa = audio_model.predict(Xa, batch_size=32).ravel()

# =========================
# FUSION WEIGHT SEARCH
# =========================
alphas = np.arange(0.0, 1.05, 0.05)
best_acc = 0
best_alpha = 0

results = []

for alpha in alphas:
    fused = alpha * pv + (1 - alpha) * pa
    y_pred = (fused > 0.5).astype(int)
    acc = accuracy_score(yv, y_pred)
    results.append((alpha, acc))

    if acc > best_acc:
        best_acc = acc
        best_alpha = alpha

# =========================
# PRINT RESULTS
# =========================
print("\nFusion weight tuning results:")
for a, acc in results:
    print(f"alpha={a:.2f} → accuracy={acc:.4f}")

print("\n✅ BEST RESULT")
print(f"Best alpha = {best_alpha:.2f}")
print(f"Best fusion accuracy = {best_acc:.4f}")

# =========================
# SAVE PLOT
# =========================
alphas_plot = [r[0] for r in results]
acc_plot = [r[1] for r in results]

plt.figure()
plt.plot(alphas_plot, acc_plot, marker='o')
plt.xlabel("Fusion weight (alpha for video)")
plt.ylabel("Accuracy")
plt.title("Audio–Video Fusion Weight Tuning")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "fusion_weight_tuning.png"))
plt.close()

print(f"\n📁 Saved fusion tuning plot to: {OUTPUT_DIR}")
