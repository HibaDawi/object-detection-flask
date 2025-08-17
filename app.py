import os, io, base64
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# ---------- Configuration ----------
WEIGHTS_PATH = "models/catdog_yolov8n.pt"  # your renamed best.pt
CLASSES_TXT = "models/classes.txt"         # optional override
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_CONTENT_MB = 10  # reject giant uploads to keep CPU responsive

# ---------- App Setup ----------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

# Load model once at startup (CPU by default; we will still force CPU during predict)
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Model not found at {WEIGHTS_PATH}. Place your .pt there.")

model = YOLO(WEIGHTS_PATH)

# Prefer classes.txt (one per line) if present, else use model.names
if os.path.exists(CLASSES_TXT):
    with open(CLASSES_TXT, "r", encoding="utf-8") as f:
        CLASSES = [ln.strip() for ln in f if ln.strip()]
else:
    # model.names is a dict like {0: "cat", 1: "dog"}
    CLASSES = [model.names[i] for i in range(len(model.names))]

def _is_allowed(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTS

def _pil_to_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _infer_and_annotate(pil_img: Image.Image, conf=0.25, iou=0.5):
    """
    Runs YOLO inference on a PIL image (forced CPU).
    Returns (detections_list, annotated_image_b64)
    """
    # Convert to RGB numpy
    img = np.array(pil_img.convert("RGB"))

    # Optional: downscale very large images to speed up CPU inference
    max_side = 960
    h, w = img.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Explicit CPU inference for full safety on free hosts
    results = model.predict(img, conf=conf, iou=iou, device="cpu", verbose=False, fuse=False, imgsz=768)
    r = results[0]

    detections = []
    # r.boxes: [xyxy, conf, cls]
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf_v = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class_id": cls_id,
            "class_name": model.names.get(cls_id, str(cls_id)),
            "confidence": round(conf_v, 3),
            "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
        })

    # Annotated preview
    annotated = r.plot()  # BGR numpy
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    pil_annotated = Image.fromarray(annotated)
    b64 = _pil_to_b64(pil_annotated)
    return detections, b64

# ---------- Routes ----------
@app.get("/")
def index():
    # Main upload UI
    return render_template("index.html", classes=CLASSES, max_mb=MAX_CONTENT_MB)

@app.post("/predict")
def predict():
    # Validate file
    if "file" not in request.files or request.files["file"].filename == "":
        return render_template("index.html", error="Please choose an image file.", classes=CLASSES, max_mb=MAX_CONTENT_MB), 400

    file = request.files["file"]
    if not _is_allowed(file.filename):
        return render_template("index.html", error="Only .jpg, .jpeg, .png, .webp are allowed.", classes=CLASSES, max_mb=MAX_CONTENT_MB), 400

    # Read into PIL
    try:
        pil_img = Image.open(file.stream)
    except Exception:
        return render_template("index.html", error="Invalid or corrupted image file.", classes=CLASSES, max_mb=MAX_CONTENT_MB), 400

    # Inference (forced CPU)
    detections, b64_img = _infer_and_annotate(pil_img)

    # Render result page
    return render_template("result.html", detections=detections, img_b64=b64_img, classes=CLASSES)

# Optional JSON API
@app.post("/api/predict")
def api_predict():
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["file"]
    if not _is_allowed(file.filename):
        return jsonify({"error": "Unsupported file type."}), 400

    try:
        pil_img = Image.open(file.stream)
    except Exception:
        return jsonify({"error": "Invalid image file."}), 400

    detections, b64_img = _infer_and_annotate(pil_img)
    return jsonify({"detections": detections, "annotated_image_b64": b64_img})

# Healthcheck (useful for hosts)
@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": True, "classes": CLASSES})

if __name__ == "__main__":
    # Keep this port for local tests; hosting will override with env var
    port = int(os.environ.get("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=False)

