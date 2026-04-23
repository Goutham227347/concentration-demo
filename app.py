from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import sqlite3
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

# ---------------- Load Model ----------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- FastAPI Init ----------------
app = FastAPI()

# Enable CORS for Chrome Extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can replace "*" with ["chrome-extension://<your-extension-id>"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Database Setup ----------------
DB_PATH = "stats.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute("""
    CREATE TABLE IF NOT EXISTS usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT,
        classification TEXT,
        timestamp TEXT
    )
""")
conn.commit()

# ---------------- Request Schema ----------------
class UrlRequest(BaseModel):
    url: str
    title: str | None = None  # ✅ optional


# ---------------- Endpoints ----------------
@app.post("/classify")
def classify_url(request: UrlRequest):
    try:
        # Combine title and url for vectorization (matching train.py)
        text = (request.title or "") + " " + request.url
        X_vec = vectorizer.transform([text])

        prediction = model.predict(X_vec)[0].lower()
        if prediction not in ["productive", "distracting"]:
          prediction = "productive"  # safe default


        # Log into DB
        c.execute(
            "INSERT INTO usage (url, classification, timestamp) VALUES (?, ?, ?)",
            (request.url, prediction, datetime.now().isoformat())
        )
        conn.commit()

        return {
            "url": request.url,
            "classification": prediction
        }
    except Exception as e:
        return {"error": str(e)}
@app.get("/stats")
def get_stats():
    # Total visits (all rows)
    c.execute("SELECT classification, COUNT(*) FROM usage GROUP BY classification")
    total = dict(c.fetchall())

    # Unique sites (distinct URLs)
    c.execute("SELECT classification, COUNT(DISTINCT url) FROM usage GROUP BY classification")
    unique = dict(c.fetchall())

    return {"total": total, "unique": unique}
@app.get("/analytics")
def get_analytics():
    """
    Returns daily productivity trend: how many productive vs distracting per day
    """
    c.execute("""
        SELECT DATE(timestamp), classification, COUNT(*) 
        FROM usage 
        GROUP BY DATE(timestamp), classification
        ORDER BY DATE(timestamp)
    """)
    rows = c.fetchall()

    data = {}
    for day, label, count in rows:
        if day not in data:
            data[day] = {"productive": 0, "distracting": 0}

        # Handle potential legacy "distractive" label
        clean_label = "distracting" if label == "distractive" else label
        if clean_label in data[day]:
            data[day][clean_label] += count

    return data


@app.get("/")
def root():
    return {"status": "Concentration Couch backend running 🚀"}

