import os
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# -----------------------------
# PATHS
# -----------------------------
VIDEO_DIR = r"results\video_new"
AUDIO_DIR = r"results\audio_new"
OUTPUT_DIR = r"results\comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD PREDICTIONS
# -----------------------------
video_y = np.load(os.path.join(VIDEO_DIR, "y_test.npy")).astype(int)
video_prob = np.load(os.path.join(VIDEO_DIR, "y_prob.npy")).ravel()

audio_y = np.load(os.path.join(AUDIO_DIR, "y_test.npy")).astype(int)
audio_prob = np.load(os.path.join(AUDIO_DIR, "y_prob.npy")).ravel()

# -----------------------------
# ALIGN LENGTHS
# -----------------------------
n = min(len(video_y), len(audio_y))

video_y = video_y[:n]
video_prob = video_prob[:n]

audio_y = audio_y[:n]
audio_prob = audio_prob[:n]

# Use video labels as reference
y_true = video_y

print("Using aligned samples:", n)
print("Video labels distribution:", np.bincount(video_y))
print("Audio labels distribution:", np.bincount(audio_y))

# -----------------------------
# METRICS
# -----------------------------
def evaluate(name, y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "Fusion Method": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob)
    }

# -----------------------------
# FUSION METHODS
# -----------------------------
results = []

# 1 Average Fusion
avg_prob = (video_prob + audio_prob) / 2
results.append(evaluate("Average Fusion", y_true, avg_prob))

# 2 Weighted Fusion: video 60%, audio 40%
weighted_60_40 = (0.6 * video_prob) + (0.4 * audio_prob)
results.append(evaluate("Weighted Fusion 60V-40A", y_true, weighted_60_40))

# 3 Weighted Fusion: video 70%, audio 30%
weighted_70_30 = (0.7 * video_prob) + (0.3 * audio_prob)
results.append(evaluate("Weighted Fusion 70V-30A", y_true, weighted_70_30))

# 4 Weighted Fusion: video 40%, audio 60%
weighted_40_60 = (0.4 * video_prob) + (0.6 * audio_prob)
results.append(evaluate("Weighted Fusion 40V-60A", y_true, weighted_40_60))

# 5 Majority / OR Voting
video_pred = (video_prob >= 0.5).astype(int)
audio_pred = (audio_prob >= 0.5).astype(int)

voting_pred = ((video_pred + audio_pred) >= 1).astype(int)
voting_prob = voting_pred.astype(float)

results.append(evaluate("Voting Fusion", y_true, voting_prob))

# -----------------------------
# SAVE
# -----------------------------
df = pd.DataFrame(results)

csv_path = os.path.join(OUTPUT_DIR, "fusion_model_comparison.csv")
df.to_csv(csv_path, index=False)

print("\nDone.")
print(df)
print("Saved to:", csv_path)