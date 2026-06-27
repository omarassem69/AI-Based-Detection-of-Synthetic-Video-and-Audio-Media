import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# PATHS
# =========================
DATA_PATH = r"C:\Users\omara\ai_synthetic_detection\data\AI_Human.csv"
MODEL_DIR = r"C:\Users\omara\ai_synthetic_detection\models"

MODEL_PATH = os.path.join(MODEL_DIR, "text_detector_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "text_vectorizer.pkl")

# =========================
# COLUMNS
# =========================
TEXT_COLUMN = "text"
LABEL_COLUMN = "generated"  # 0 = human, 1 = AI/generated

# =========================
# LOAD DATA
# =========================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, usecols=[TEXT_COLUMN, LABEL_COLUMN])

print("Columns:", df.columns)
print("Shape before cleaning:", df.shape)

df = df.dropna()
df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

print("Shape after cleaning:", df.shape)
print("Label distribution:")
print(df[LABEL_COLUMN].value_counts())

# =========================
# SPLIT
# =========================
X = df[TEXT_COLUMN]
y = df[LABEL_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# TF-IDF
# =========================
print("Building TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# MODEL
# =========================
print("Training Logistic Regression text detector...")
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_vec, y_train)

# =========================
# EVALUATION
# =========================
print("Evaluating model...")
y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print("\n✅ Text model saved successfully!")
print("Model path:", MODEL_PATH)
print("Vectorizer path:", VECTORIZER_PATH)