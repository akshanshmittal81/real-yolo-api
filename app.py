from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import os
import logging
from PIL import Image

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Model Loading ────────────────────────────────────────────────────────────
# Loaded once at startup to avoid reloading on every request
model = None

def load_model():
    global model
    try:
        from ultralytics import YOLO
        # YOLOv8 pretrained on COCO — detects cars, trucks etc.
        # For car-damage-specific model, swap with:
        #   YOLO("path/to/your/best.pt")
        # or a HuggingFace car damage model
        model = YOLO("yolov8m.pt")  # auto-downloads on first run (~52MB)
        logger.info("✅ YOLO model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Model load failed: {e}")
        model = None

# Car/vehicle related COCO class IDs
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Parts we map detections to (simulated from bounding box regions)
DAMAGE_PARTS = ["bumper", "hood", "door", "fender",
                "windshield", "headlight", "taillight", "trunk"]

def map_to_car_parts(boxes, img_width, img_height):
    """Map detected bounding boxes to car part names by position."""
    parts = set()
    for box in boxes:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2 / img_width   # normalized center x
        cy = (y1 + y2) / 2 / img_height   # normalized center y

        # Top half of image → hood / windshield / roof
        if cy < 0.4:
            parts.add("hood" if cx < 0.5 else "windshield")
        # Bottom region → bumper
        elif cy > 0.7:
            parts.add("bumper")
        # Left side
        elif cx < 0.3:
            parts.add("headlight" if cy < 0.55 else "door")
        # Right side
        elif cx > 0.7:
            parts.add("taillight" if cy < 0.55 else "fender")
        else:
            parts.add("door")

    return list(parts) if parts else ["bumper"]  # default if no boxes

def severity_from_confidence(conf: float) -> str:
    if conf >= 0.80:
        return "high"
    elif conf >= 0.55:
        return "medium"
    return "low"


def analyze_image(image_base64: str):
    img_bytes = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_w, img_h = img.size

    if model is None:
        return {"error": "Model not loaded"}, 503

    results = model(img, verbose=False)[0]

    vehicle_boxes = []
    max_conf = 0.0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id in VEHICLE_CLASSES and conf > 0.3:
            coords = box.xyxy[0].tolist()
            vehicle_boxes.append(coords)
            max_conf = max(max_conf, conf)

    damage_detected = len(vehicle_boxes) > 0
    damaged_parts = []
    severity = "none"
    confidence = 0.0

    if damage_detected:
        damaged_parts = map_to_car_parts(vehicle_boxes, img_w, img_h)
        confidence = round(max_conf, 2)
        severity = severity_from_confidence(max_conf)

    return {
        "damage_detected": damage_detected,
        "damaged_parts": damaged_parts,
        "severity": severity,
        "confidence": confidence
    }, 200


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "service": "YOLO Vehicle Damage Detection API",
        "model": "YOLOv8m (COCO pretrained)",
        "version": "2.0.0"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "image_base64" not in data:
            return jsonify({"error": "Missing field: image_base64"}), 400

        try:
            img_bytes = base64.b64decode(data["image_base64"])
            if len(img_bytes) < 100:
                return jsonify({"error": "Image too small or invalid"}), 400
        except Exception:
            return jsonify({"error": "Invalid base64 image"}), 400

        result, status = analyze_image(data["image_base64"])
        return jsonify(result), status

    except Exception as e:
        logger.error(f"Predict error: {e}")
        return jsonify({"error": str(e)}), 500


# ─── Startup ─────────────────────────────────────────────────────────────────
with app.app_context():
    load_model()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
