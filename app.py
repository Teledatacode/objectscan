from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- importa CORS
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

        # --- Aquí podemos centrar el objeto ---
        # Por simplicidad, hacemos un recorte central (placeholder)
        h, w, _ = img_np.shape
        crop_size = min(h,w)
        start_h = (h - crop_size)//2
        start_w = (w - crop_size)//2
        img_crop = img_np[start_h:start_h+crop_size, start_w:start_w+crop_size]

        # Convertir de vuelta a base64
        pil_crop = Image.fromarray(img_crop)
        buffer = io.BytesIO()
        pil_crop.save(buffer, format="JPEG")
        processed_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        processed.append({
            "image": processed_b64,
            "vector": cap.get("vector")
        })

    return jsonify({"captures": processed})

if __name__ == "__main__":
    app.run(debug=True)
