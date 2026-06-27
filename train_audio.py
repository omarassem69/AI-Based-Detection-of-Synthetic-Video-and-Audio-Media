import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# PATHS
# =========================
X_PATH = r"C:\Users\omara\ai_synthetic_detection\data\output\audio\X_cvoicefake_en.npy"
Y_PATH = r"C:\Users\omara\ai_synthetic_detection\data\output\audio\y_cvoicefake_en.npy"
RESULT_DIR = r"C:\Users\omara\ai_synthetic_detection\data\output\result audio"

os.makedirs(RESULT_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
print("Loading audio features...")
X = np.load(X_PATH)
y = np.load(Y_PATH)

print("X shape:", X.shape)
print("y shape:", y.shape)

X = X[..., np.newaxis]  # add channel dim

# =========================
# SPLIT DATA
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)

# =========================
# MODEL
# =========================
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(40, 400, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN (NO CHECKPOINT)
# =========================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32
)

# =========================
# SAVE MODEL (SAFE)
# =========================
model.save(os.path.join(RESULT_DIR, "audio_model_final.keras"))

# =========================
# EVALUATION
# =========================
print("\nEvaluating on test set...")
y_pred = (model.predict(X_test) > 0.5).astype("int32")

report = classification_report(y_test, y_pred, target_names=["Bonafide", "Fake"])
print(report)

with open(os.path.join(RESULT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Bonafide", "Fake"],
            yticklabels=["Bonafide", "Fake"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Audio Confusion Matrix")
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))
plt.close()

# =========================
# TRAINING CURVES
# =========================
plt.figure()
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend()
plt.title("Audio Accuracy")
plt.savefig(os.path.join(RESULT_DIR, "accuracy.png"))
plt.close()

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Audio Loss")
plt.savefig(os.path.join(RESULT_DIR, "loss.png"))
plt.close()

print("\n✅ AUDIO MODEL TRAINING COMPLETE")
print("📁 Results saved to:", RESULT_DIR)
