from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import requests
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ CORS (allow frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Google Drive Links
MODEL_URL = "https://drive.google.com/uc?export=download&id=1o3U2SSwW7jrMLYVY6-NndltO7yOa-l00"
VECTORIZER_URL = "https://drive.google.com/uc?export=download&id=1KiJ7uOmbPbZttt-UeyaO6Vn_1kuJE7wI"
BEHAVIOR_URL = "https://drive.google.com/uc?export=download&id=1davRWvD6N5bvswZ-9jj6uaf0-r1Ikx-2"

# ✅ Download function
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        r = requests.get(url)
        with open(filename, "wb") as f:
            f.write(r.content)
        print(f"{filename} downloaded")

# ✅ Download models
download_file(MODEL_URL, "model.pkl")
download_file(VECTORIZER_URL, "vectorizer.pkl")
download_file(BEHAVIOR_URL, "behavior_model.pkl")

# ✅ Load models
print("Loading models...")
text_model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
behavior_model = joblib.load("behavior_model.pkl")
print("Models loaded successfully")

# 📥 Input Schema
class ReviewInput(BaseModel):
    review: str
    reviews_per_user: float
    avg_label: float
    time_diff: float
    is_duplicate: float

# 🔥 RULE-BASED SMART FEATURE FUNCTION
def smart_text_features(text):
    text_lower = text.lower()

    exclamations = text.count("!")
    repeated_words = len(text.split()) - len(set(text.split()))
    caps_ratio = sum(1 for c in text if c.isupper()) / (len(text) + 1)

    score = 0

    if exclamations > 3:
        score += 0.3
    if repeated_words > 2:
        score += 0.3
    if caps_ratio > 0.3:
        score += 0.2
    if "amazing" in text_lower or "best" in text_lower or "wow" in text_lower:
        score += 0.2

    return min(score, 1.0)

# 🏠 Home route
@app.get("/")
def home():
    return {"message": "Fake Review Detection Backend is running 🚀"}

# 🧠 Prediction API
@app.post("/predict")
def predict(data: ReviewInput):
    try:
        # ✅ TEXT MODEL
        text_vector = vectorizer.transform([data.review])
        text_prob = text_model.predict_proba(text_vector)[0][1]

        # ✅ BEHAVIOR MODEL
        behavior_features = np.array([[
            data.reviews_per_user,
            data.avg_label,
            data.time_diff,
            data.is_duplicate
        ]])
        behavior_prob = behavior_model.predict_proba(behavior_features)[0][1]

        # ✅ RULE-BASED SCORE
        rule_score = smart_text_features(data.review)

        # ✅ FINAL SCORE (Improved)
        final_score = (
            0.5 * text_prob +
            0.3 * behavior_prob +
            0.2 * rule_score
        )

        # ✅ Prediction
        prediction = "Fake" if final_score > 0.5 else "Genuine"

        # ✅ Explainability (for viva 🔥)
        reasons = []

        if rule_score > 0.3:
            reasons.append("Suspicious text pattern")

        if data.reviews_per_user > 20:
            reasons.append("Too many reviews by user")

        if data.is_duplicate == 1:
            reasons.append("Duplicate review detected")

        if data.time_diff < 2:
            reasons.append("Reviews posted too quickly")

        # ✅ Response
        return {
            "prediction": prediction,
            "text_score": float(text_prob),
            "behavior_score": float(behavior_prob),
            "rule_score": float(rule_score),
            "final_score": float(final_score),
            "reasons": reasons
        }

    except Exception as e:
        return {"error": str(e)}