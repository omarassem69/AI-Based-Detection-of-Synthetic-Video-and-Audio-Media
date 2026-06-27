import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, LSTM, Reshape
from tensorflow.keras.callbacks import Callback

# -----------------------------
# SETTINGS
# -----------------------------
X_PATH = r"data\output\audio\X_cvoicefake_en.npy"
Y_PATH = r"data\output\audio\y_cvoicefake_en.npy"

OUTPUT_DIR = r"results\comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS = 3
BATCH_SIZE = 8
SAMPLE_PER_CLASS = 300

# -----------------------------
# PROGRESS CALLBACK
# -----------------------------
class TQDMProgressBar(Callback):
    def on_train_begin(self, logs=None):
        self.epoch_bar = tqdm(total=self.params["epochs"], desc="Training epochs")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch_bar.update(1)
        self.epoch_bar.set_postfix({
            "loss": f"{logs.get('loss', 0):.4f}",
            "acc": f"{logs.get('accuracy', 0):.4f}",
            "val_loss": f"{logs.get('val_loss', 0):.4f}",
            "val_acc": f"{logs.get('val_accuracy', 0):.4f}"
        })

    def on_train_end(self, logs=None):
        self.epoch_bar.close()

# -----------------------------
# LOAD DATA LIGHTLY
# -----------------------------
print("Loading labels...")
y_all = np.load(Y_PATH)

print("Memory mapping X, not loading full file...")
X_all = np.load(X_PATH, mmap_mode="r")

print("Original X shape:", X_all.shape)
print("Original y shape:", y_all.shape)

real_idx = np.where(y_all == 0)[0]
fake_idx = np.where(y_all == 1)[0]

np.random.seed(42)
real_sample = np.random.choice(real_idx, size=min(SAMPLE_PER_CLASS, len(real_idx)), replace=False)
fake_sample = np.random.choice(fake_idx, size=min(SAMPLE_PER_CLASS, len(fake_idx)), replace=False)

selected_idx = np.concatenate([real_sample, fake_sample])
np.random.shuffle(selected_idx)

print("Loading only selected sample...")
X_list = []

for idx in tqdm(selected_idx, desc="Loading selected audio samples"):
    X_list.append(X_all[idx])

X = np.array(X_list, dtype="float32")
y = y_all[selected_idx].astype(int)

if X.max() > 1:
    X = X / 255.0

print("Sample X shape:", X.shape)
print("Sample y distribution:", np.bincount(y))

if len(X.shape) == 3:
    X_cnn = np.expand_dims(X, axis=-1)
else:
    X_cnn = X

print("CNN input shape:", X_cnn.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X_cnn,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_flat = X.reshape((X.shape[0], -1))

X_train_flat, X_test_flat, y_train_flat, y_test_flat = train_test_split(
    X_flat,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# METRICS
# -----------------------------
def calc_metrics(name, y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob)
    }

# -----------------------------
# DEEP LEARNING MODELS
# -----------------------------
def build_audio_cnn(input_shape):
    return Sequential([
        Input(shape=input_shape),
        Conv2D(16, (3, 3), activation="relu"),
        MaxPooling2D(),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

def build_audio_lstm(input_shape):
    freq = input_shape[0]
    time = input_shape[1]

    return Sequential([
        Input(shape=input_shape),
        Reshape((time, freq)),
        LSTM(64),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

def evaluate_dl_model(model, name):
    print(f"\n==============================")
    print(f"Training {name}")
    print(f"==============================")

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[TQDMProgressBar()]
    )

    print(f"Predicting {name}...")
    y_prob = model.predict(X_test, verbose=1).ravel()

    result = calc_metrics(name, y_test, y_prob)
    print("Result:", result)
    return result

# -----------------------------
# MACHINE LEARNING MODELS
# -----------------------------
def evaluate_logistic_regression():
    print("\n==============================")
    print("Training Logistic Regression")
    print("==============================")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    for _ in tqdm(range(1), desc="Training Logistic Regression"):
        model.fit(X_train_flat, y_train_flat)

    y_prob = model.predict_proba(X_test_flat)[:, 1]

    result = calc_metrics(
        "Logistic Regression",
        y_test_flat,
        y_prob
    )

    print("Result:", result)
    return result

def evaluate_random_forest():
    print("\n==============================")
    print("Training Random Forest")
    print("==============================")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    for _ in tqdm(range(1), desc="Training Random Forest"):
        model.fit(X_train_flat, y_train_flat)

    y_prob = model.predict_proba(X_test_flat)[:, 1]

    result = calc_metrics(
        "Random Forest",
        y_test_flat,
        y_prob
    )

    print("Result:", result)
    return result

# -----------------------------
# RUN COMPARISON
# -----------------------------
all_results = []

input_shape = X_train.shape[1:]

models_to_run = [
    ("Audio CNN", lambda: evaluate_dl_model(build_audio_cnn(input_shape), "Audio CNN")),
    ("LSTM", lambda: evaluate_dl_model(build_audio_lstm(input_shape), "LSTM")),
    ("Logistic Regression", evaluate_logistic_regression),
    ("Random Forest", evaluate_random_forest),
]

for model_name, runner in tqdm(models_to_run, desc="Overall audio comparison"):
    result = runner()
    all_results.append(result)
    tf.keras.backend.clear_session()

# -----------------------------
# SAVE RESULTS
# -----------------------------
df = pd.DataFrame(all_results)

csv_path = os.path.join(OUTPUT_DIR, "audio_model_comparison.csv")
df.to_csv(csv_path, index=False)

print("\nDone.")
print(df)
print("Saved to:", csv_path)