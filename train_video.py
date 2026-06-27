import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.utils import resample
import os

# =========================
# CONFIG
# =========================
X_PATH = r"data/output/video/X_ffpp.npy"
Y_PATH = r"data/output/video/y_ffpp.npy"
SAVE_DIR = r"models"
IMG_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS_FROZEN = 5
EPOCHS_FINE = 5
SEED = 42

os.makedirs(SAVE_DIR, exist_ok=True)
np.random.seed(SEED)

# =========================
# LOAD DATA
# =========================
print("Loading video features...")
X = np.load(X_PATH)
y = np.load(Y_PATH)

print("Original distribution:", dict(zip(*np.unique(y, return_counts=True))))

# =========================
# BALANCE DATA (CRITICAL)
# =========================
X_real = X[y == 0]
y_real = y[y == 0]

X_fake = X[y == 1]
y_fake = y[y == 1]

N = min(len(X_real), len(X_fake))

X_real = resample(X_real, n_samples=N, random_state=SEED)
X_fake = resample(X_fake, n_samples=N, random_state=SEED)

y_real = np.zeros(N)
y_fake = np.ones(N)

X = np.concatenate([X_real, X_fake])
y = np.concatenate([y_real, y_fake])

# Shuffle
idx = np.random.permutation(len(y))
X, y = X[idx], y[idx]

print("Balanced distribution:", dict(zip(*np.unique(y, return_counts=True))))

# =========================
# SPLIT
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# =========================
# MODEL
# =========================
base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
out = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN (FROZEN)
# =========================
print("\n🔒 Training frozen backbone...")
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_FROZEN,
    batch_size=BATCH_SIZE,
    verbose=1
)

# =========================
# FINE-TUNE
# =========================
print("\n🔓 Fine-tuning top layers...")
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_FINE,
    batch_size=BATCH_SIZE,
    verbose=1
)

# =========================
# SAVE
# =========================
model.save(os.path.join(SAVE_DIR, "video_model_balanced.keras"))
print("\n✅ Video model saved to models/video_model_balanced.keras")
