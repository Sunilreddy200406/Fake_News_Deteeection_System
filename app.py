import pickle
from pathlib import Path

from flask_cors import CORS
from flask import Flask, jsonify, render_template, request, send_file

from src.history_db import clear_history, export_history_to_csv, fetch_history, init_db, save_history
from src.hybrid_service import analyze_news
from src.preprocess import ensure_nltk_resources


app = Flask(__name__)
CORS(app)
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"

model_bundle = None


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found at models/best_model.pkl. Run `python -m src.train` first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@app.before_request
def setup():
    global model_bundle
    if model_bundle is None:
        ensure_nltk_resources()
        model_bundle = load_model()
        init_db()


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def _handle_analysis_request():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text") or request.form.get("text", "")
    source_url = payload.get("source_url") or request.form.get("source_url", "")
    text = text.strip()
    source_url = source_url.strip()

    if not text:
        return jsonify({"error": "Please provide news text."}), 400

    result = analyze_news(text=text, source_url=source_url, model_bundle=model_bundle)
    save_history(
        news_text=text,
        source_url=source_url,
        result=result.get("result", "Unverified"),
        method=result.get("verification_method", "Machine Learning"),
    )
    return jsonify(result)


@app.route("/analyze", methods=["POST"])
def analyze():
    return _handle_analysis_request()


@app.route("/predict", methods=["POST"])
def predict():
    # Alias endpoint for compatibility with earlier versions.
    return _handle_analysis_request()


@app.route("/history", methods=["GET"])
def history():
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1
    try:
        limit = max(1, min(100, int(request.args.get("limit", 10))))
    except ValueError:
        limit = 10

    result_filter = (request.args.get("result", "") or "").strip()
    if result_filter not in {"", "Real", "Fake", "Unverified"}:
        result_filter = ""
    offset = (page - 1) * limit
    items = fetch_history(limit=limit, offset=offset, result_filter=result_filter or None)
    return jsonify({"items": items, "page": page, "limit": limit, "result_filter": result_filter or None})


@app.route("/history", methods=["DELETE"])
def history_clear():
    clear_history()
    return jsonify({"message": "History cleared."})


@app.route("/history/export", methods=["GET"])
def history_export():
    export_path = ROOT / "data" / "history_export.csv"
    csv_file = export_history_to_csv(export_path)
    return send_file(csv_file, as_attachment=True, download_name="history_export.csv", mimetype="text/csv")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
