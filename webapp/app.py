"""
Flask web app:  Input message -> Neural network -> Spam / Not Spam  (+ sentiment, topics, dashboard)

Run locally :  python webapp/app.py            -> http://127.0.0.1:8080
Production  :  gunicorn -b :$PORT webapp.app:app      (see Dockerfile)
"""
import json
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from reviewintel.inference import Predictor  # noqa: E402

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

try:
    PREDICTOR = Predictor(str(ROOT / "models"))
except FileNotFoundError:
    PREDICTOR = None       # models not trained yet -> pages show a friendly message


def _load(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


EXAMPLES = [
    ("Genuine - positive", "Delivery was super fast and the battery easily lasts a full day. Camera is sharp too. Totally paisa vasool!"),
    ("Genuine - negative", "Very disappointed. The box was torn and the fabric is rough. Refund still not received after three weeks."),
    ("Hinglish", "Packing bahut kharab thi aur delivery late ho gayi, quality bhi bekaar hai."),
    ("Spam - promo link", "Get 90% OFF on all mobiles today only! Visit www.dealzmart-deals.in or WhatsApp 9876501234"),
    ("Fake / paid review", "Seller is giving Rs 100 refund for a 5 star review. DM me for details."),
    ("Fake praise", "Best product ever!!! Must buy!!! 5 star 5 star 5 star"),
]


@app.route("/")
def index():
    return render_template("index.html", examples=EXAMPLES, ready=PREDICTOR is not None, meta=_load(ROOT / "models" / "meta.json", {}))


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", d=_load(ROOT / "webapp" / "dashboard.json", None))


@app.route("/model")
def model_page():
    return render_template("model.html", r=_load(ROOT / "outputs" / "results.json", None))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if PREDICTOR is None:
        return jsonify(error="Model not found. Run  python run_pipeline.py  first."), 503
    data = request.get_json(silent=True) or {}
    text = data.get("text") or request.form.get("text", "")
    try:
        return jsonify(PREDICTOR.predict(text))
    except ValueError as e:
        return jsonify(error=str(e)), 400


@app.route("/health")
def health():
    return jsonify(status="ok", model_loaded=PREDICTOR is not None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
