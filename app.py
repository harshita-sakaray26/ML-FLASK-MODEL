# app.py (cleaned & fixed)
import os
import io
import base64
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient, DESCENDING
from bson import ObjectId
from dotenv import load_dotenv
from datetime import datetime
from dateutil import parser as dateparser 

# load .env from current working directory
load_dotenv()

# ---------- config from .env ----------
MONGO_URI = os.getenv("MONGODB_URI")            # your .env has MONGODB_URI
DATABASE_NAME = os.getenv("DATABASE_NAME")      # your .env has DATABASE_NAME
MODEL_API_URL = os.getenv("MODEL_API_URL")      # your .env has MODEL_API_URL

# quick validation (fail fast with clear message)
if not MONGO_URI:
    raise RuntimeError("MONGODB_URI not set in environment or .env")
if not DATABASE_NAME:
    raise RuntimeError("DATABASE_NAME not set in environment or .env")
if not MODEL_API_URL:
    raise RuntimeError("MODEL_API_URL not set in environment or .env")

# OPTIONAL: debug print (comment out in production)
print("Using DATABASE_NAME =", DATABASE_NAME)
print("Using MODEL_API_URL =", MODEL_API_URL[:80] + "..." if len(MODEL_API_URL) > 80 else MODEL_API_URL)

# ---------- flask & mongodb ----------
app = Flask(__name__)
CORS(app)

mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[DATABASE_NAME]

def _convert_bson(o):
    if isinstance(o, ObjectId):
        return str(o)
    if isinstance(o, dict):
        return {k: _convert_bson(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_convert_bson(i) for i in o]
    return o

@app.route("/")
def home():
    return jsonify({"status": "Server is running", "database": DATABASE_NAME})

@app.route("/api/iot_device", methods=["POST"])
def api_iot_device():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "JSON body required"}), 400

    device_id = payload.get("device_id")
    if not device_id:
        return jsonify({"error": "device_id is required"}), 400

    prediction_url = payload.get("prediction_url") or MODEL_API_URL

    try:
        coll = mongo_db[device_id]
    except Exception as e:
        return jsonify({"error": f"Cannot access collection '{device_id}': {str(e)}"}), 500

    # --- Find most recent doc explicitly ---
    # Prefer documents that actually have 'timestamp' field (descending),
    # else fallback to created_at or _id
    device_doc = None
    # try timestamp first (only docs that have it)
    device_doc = coll.find_one({"timestamp": {"$exists": True}}, sort=[("timestamp", DESCENDING)])
    if not device_doc:
        device_doc = coll.find_one({"created_at": {"$exists": True}}, sort=[("created_at", DESCENDING)])
    if not device_doc:
        device_doc = coll.find_one(sort=[("_id", DESCENDING)])  # last resort

    if not device_doc:
        return jsonify({"error": f"No documents found for device collection: {device_id}"}), 404

    device_doc_safe = _convert_bson(device_doc)

    # --- Extract a timestamp value (if present) and normalise to ISO8601 ---
    def _extract_timestamp(doc):
        # common top-level fields
        for k in ("timestamp", "time", "created_at", "received_at"):
            if k in doc and doc[k]:
                return k, doc[k]
        # shallow nested search (one level)
        for k, v in doc.items():
            if isinstance(v, dict):
                for subk in ("timestamp", "time", "created_at", "received_at"):
                    if subk in v and v[subk]:
                        return f"{k}.{subk}", v[subk]
        return None, None

    ts_field, ts_value = _extract_timestamp(device_doc_safe)
    device_timestamp_iso = None
    device_timestamp_raw = None
    if ts_value is not None:
        device_timestamp_raw = ts_value
        # If it's a datetime object (BSON datetime), convert to ISO
        try:
            if hasattr(ts_value, "isoformat"):
                device_timestamp_iso = ts_value.isoformat()
            else:
                # try parsing if it's a string like "2025-11-28T10:00:00Z" or "2025-11-28 10:00:00"
                device_timestamp_iso = dateparser.parse(str(ts_value)).isoformat()
        except Exception:
            # fallback: store string repr
            device_timestamp_iso = str(ts_value)

    # locate image (same logic as before) ...
    image_data = None
    image_field_name = None
    candidates = ("image", "image_base64", "image_url", "img", "photo", "picture")
    for c in candidates:
        if c in device_doc_safe and device_doc_safe[c]:
            image_data = device_doc_safe[c]
            image_field_name = c
            break
    if not image_data:
        for k, v in device_doc_safe.items():
            if isinstance(v, dict):
                for c in candidates:
                    if c in v and v[c]:
                        image_data = v[c]
                        image_field_name = f"{k}.{c}"
                        break
            if image_data:
                break
    if not image_data:
        return jsonify({
            "error": "No image found in device document. checked keys: " + ", ".join(candidates),
            "device_id": device_id,
            "device_data": device_doc_safe
        }), 400

    # send to model API (same as your previous logic) ...
    headers = {"Accept": "application/json"}
    try:
        if isinstance(image_data, str) and image_data.lower().startswith(("http://", "https://")):
            pred_resp = requests.post(prediction_url, json={"image_url": image_data}, headers=headers, timeout=30)
        elif isinstance(image_data, str) and (image_data.startswith("data:image/") or _looks_like_base64(image_data.split(",")[-1])):
            b64str = image_data.split(",")[-1]
            img_bytes = base64.b64decode(b64str)
            files = {"file": ("image.jpg", io.BytesIO(img_bytes), "image/jpeg")}
            pred_resp = requests.post(prediction_url, files=files, headers=headers, timeout=30)
        else:
            pred_resp = requests.post(prediction_url, json={"image": image_data}, headers=headers, timeout=30)
        pred_resp.raise_for_status()
    except requests.exceptions.RequestException as re:
        return jsonify({
            "error": "Prediction API request failed",
            "details": str(re),
            "device_id": device_id,
            "device_data": device_doc_safe
        }), 502

    try:
        pred_json = pred_resp.json()
    except Exception:
        pred_json = {"raw_text": pred_resp.text}

    main_prediction = _extract_main_prediction(pred_json)

    response = {
        "device_id": device_id,
        "image_field": image_field_name,
        "device_data": device_doc_safe,
        # timestamp fields added for clarity
        "device_timestamp_field": ts_field,         # e.g. "timestamp" or "sensor.timestamp"
        "device_timestamp_raw": device_timestamp_raw,
        "device_timestamp_iso": device_timestamp_iso,  # ISO8601 string or None
        "model_prediction": main_prediction,
        "prediction_api_response": pred_json
    }

    return jsonify(response), 200

def _looks_like_base64(s: str) -> bool:
    if not isinstance(s, str) or len(s) < 20:
        return False
    try:
        base64.b64decode(s[:50], validate=True)
        return True
    except Exception:
        return False

def _extract_main_prediction(pred_json):
    if not isinstance(pred_json, dict):
        return pred_json
    keys_try = ("main_prediction", "main_pred", "prediction", "pred", "label", "result", "class")
    for k in keys_try:
        if k in pred_json:
            return pred_json[k]
    for v in pred_json.values():
        if isinstance(v, dict):
            for k in keys_try:
                if k in v:
                    return v[k]
    if "predictions" in pred_json and isinstance(pred_json["predictions"], list) and pred_json["predictions"]:
        return pred_json["predictions"][0]
    return pred_json

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
