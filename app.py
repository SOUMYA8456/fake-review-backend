from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import requests
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ CORS
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

@app.get("/")
def home():
    return {"message": "Backend is running 🚀"}

@app.post("/predict")
def predict(data: ReviewInput):
    try:
        # 🔍 DEBUG INPUT
        print("Incoming Review:", data.review)

        # ================= TEXT MODEL =================
        text_vector = vectorizer.transform([data.review])

        print("Vector shape:", text_vector.shape)

        # 🚨 If vector is empty → FIX fallback
        if text_vector.shape[1] == 0:
            text_prob = 0.5
        else:
            text_prob = text_model.predict_proba(text_vector)[0][1]

        # ================= BEHAVIOR MODEL =================
        behavior_features = np.array([[
            data.reviews_per_user,
            data.avg_label,
            data.time_diff,
            data.is_duplicate
        ]])

        print("Behavior input:", behavior_features)

        behavior_prob = behavior_model.predict_proba(behavior_features)[0][1]

        # ================= FINAL SCORE =================
        final_score = (0.7 * text_prob) + (0.3 * behavior_prob)

        prediction = "Fake" if final_score > 0.5 else "Genuine"

        return {
            "prediction": prediction,
            "text_score": round(float(text_prob), 3),
            "behavior_score": round(float(behavior_prob), 3),
            "final_score": round(float(final_score), 3)
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}