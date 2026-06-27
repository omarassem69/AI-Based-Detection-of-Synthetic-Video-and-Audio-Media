import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.callbacks import Callback

# -----------------------------
# SETTINGS
# -----------------------------
X_PATH = r"data\output\video\X_ffpp.npy"
Y_PATH = r"data\output\video\y_ffpp.npy"

OUTPUT_DIR = r"results\comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 160
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

for idx in tqdm(selected_idx, desc="Loading selected images"):
    X_list.append(X_all[idx])

X = np.array(X_list, dtype="float32") / 255.0
y = y_all[selected_idx].astype(int)

print("Sample X shape:", X.shape)
print("Sample y distribution:", np.bincount(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# EVALUATE MODEL
# -----------------------------
def evaluate_model(model, name):
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
    y_pred = (y_prob >= 0.5).astype(int)

    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_prob)
    }

    print("Result:", results)
    return results

# -----------------------------
# MODELS
# -----------------------------
def build_simple_cnn():
    return Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        Conv2D(16, (3, 3), activation="relu"),
        MaxPooling2D(),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

def build_transfer_model(base_name):
    if base_name == "MobileNetV2":
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    elif base_name == "ResNet50":
        base = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    elif base_name == "EfficientNetB0":
        base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    else:
        raise ValueError("Unknown model name")

    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)

    return Model(inputs=base.input, outputs=output)

# -----------------------------
# RUN COMPARISON
# -----------------------------
all_results = []

model_builders = [
    ("Simple CNN", build_simple_cnn),
    ("MobileNetV2", lambda: build_transfer_model("MobileNetV2")),
    ("ResNet50", lambda: build_transfer_model("ResNet50")),
    ("EfficientNetB0", lambda: build_transfer_model("EfficientNetB0")),
]

for model_name, builder in tqdm(model_builders, desc="Overall model comparison"):
    model = builder()
    result = evaluate_model(model, model_name)
    all_results.append(result)

    tf.keras.backend.clear_session()

# -----------------------------
# SAVE RESULTS
# -----------------------------
df = pd.DataFrame(all_results)

csv_path = os.path.join(OUTPUT_DIR, "video_model_comparison.csv")
df.to_csv(csv_path, index=False)

print("\nDone.")
print(df)
print("Saved to:", csv_path)