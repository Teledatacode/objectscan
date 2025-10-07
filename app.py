from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Permitir requests grandes (100 MB)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  

@app.route("/process", methods=["POST"])
def process_images():
    try:
        data = request.json
        captures = data.get("captures", [])
        processed = []

        for cap in captures:
            img_data = cap.get("image")
            if img_data.startswith("data:image"):
                img_b64 = img_data.split(",")[1]
            else:
                img_b64 = img_data  # por si viene sin prefijo
            img_bytes = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = np.array(img)

            # --- Centrar objeto (placeholder: recorte central) ---
            h, w, _ = img_np.shape
            crop_size = min(h, w)
            start_h = (h - crop_size) // 2
            start_w = (w - crop_size) // 2
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

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
