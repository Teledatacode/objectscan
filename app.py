from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

@app.route("/process", methods=["POST"])
def process_images():
    data = request.json
    captures = data.get("captures", [])

    processed = []

    for cap in captures:
        img_b64 = cap.get("image").split(",")[1]
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)

        # --- Centrar objeto (como antes) ---
        h, w, _ = img_np.shape
        crop_size = min(h, w)
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        img_crop = img_np[start_h:start_h+crop_size, start_w:start_w+crop_size]

        # --- Crear versión con fondo transparente ---
        # Convertir a HSV para segmentar fondo blanco
        hsv = cv2.cvtColor(img_crop, cv2.COLOR_RGB2HSV)
        lower = np.array([0,0,200])  # blanco aproximado
        upper = np.array([180,40,255])
        mask = cv2.inRange(hsv, lower, upper)
        mask_inv = cv2.bitwise_not(mask)

        b, g, r = cv2.split(img_crop)
        alpha = mask_inv
        img_rgba = cv2.merge([r, g, b, alpha])  # RGBA

        # Convertir a base64
        pil_img = Image.fromarray(img_rgba)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        img_transparent_b64 = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

        # También guardamos versión con fondo
        pil_img_bg = Image.fromarray(img_crop)
        buffer_bg = io.BytesIO()
        pil_img_bg.save(buffer_bg, format="JPEG")
        img_bg_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer_bg.getvalue()).decode()

        processed.append({
            "image_bg": img_bg_b64,           # con fondo
            "image_transparent": img_transparent_b64, # transparente
            "vector": cap.get("vector")
        })

    return jsonify({"captures": processed})

if __name__ == "__main__":
    app.run(debug=True)
