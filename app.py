from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import requests
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ CORS (allow frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Google Drive DIRECT download links
MODEL_URL = "https://drive.google.com/uc?id=1o3U2SSwW7jrMLYVY6-NndltO7yOa-l00"
VECTORIZER_URL = "https://drive.google.com/uc?id=1KiJ7uOmbPbZttt-UeyaO6Vn_1kuJE7wI"
BEHAVIOR_URL = "https://drive.google.com/uc?id=1davRWvD6N5bvswZ-9jj6uaf0-r1Ikx-2"

# ✅ File names (saved in Render disk)
MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
BEHAVIOR_FILE = "behavior_model.pkl"


# ✅ Robust download function
def download_file(url, filename):
    try:
        if not os.path.exists(filename):
            print(f"⬇️ Downloading {filename}...")
            response = requests.get(url, timeout=60)

            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                print(f"✅ {filename} downloaded successfully")
            else:
                raise Exception(f"❌ Failed to download {filename}")
        else:
            print(f"✅ {filename} already exists")
    except Exception as e:
        print(f"ERROR downloading {filename}: {e}")
        raise e


# ✅ Load models safely
def load_models():
    global text_model, vectorizer, behavior_model

    download_file(MODEL_URL, MODEL_FILE)
    download_file(VECTORIZER_URL, VECTORIZER_FILE)
    download_file(BEHAVIOR_URL, BEHAVIOR_FILE)

    print("🔄 Loading models...")

    text_model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    behavior_model = joblib.load(BEHAVIOR_FILE)

    print("✅ Models loaded successfully")


# 🔥 Load at startup
load_models()


# 📥 Input schema
class ReviewInput(BaseModel):
    review: str
    reviews_per_user: float
    avg_label: float
    time_diff: float
    is_duplicate: float


# 🧠 Prediction API
@app.post("/predict")
def predict(data: ReviewInput):
    try:
        # 🔹 Text Model
        text_vector = vectorizer.transform([data.review])
        text_prob = text_model.predict_proba(text_vector)[0][1]

        # 🔹 Behavior Model
        behavior_features = np.array([[
            data.reviews_per_user,
            data.avg_label,
            data.time_diff,
            data.is_duplicate
        ]])

        behavior_prob = behavior_model.predict_proba(behavior_features)[0][1]

        # 🔥 Final Score
        final_score = (0.7 * text_prob) + (0.3 * behavior_prob)

        prediction = "Fake" if final_score > 0.5 else "Genuine"

        return {
            "prediction": prediction,
            "text_score": float(text_prob),
            "behavior_score": float(behavior_prob),
            "final_score": float(final_score)
        }

    except Exception as e:
        return {
            "error": str(e)
        }


# ✅ Health check route (VERY IMPORTANT)
@app.get("/")
def home():
    return {"message": "Backend is running successfully 🚀"}