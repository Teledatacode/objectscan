from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)  # permite que tu frontend acceda al servidor

@app.route("/process", methods=["POST"])
def process_images():
    data = request.json
    captures = data.get("captures", [])

    processed = []

    for idx, cap in enumerate(captures):
        img_b64 = cap.get("image").split(",")[1]  # extraer base64
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)

        # --- Centrar objeto usando contornos ---
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Bounding box del objeto
            x, y, w, h = cv2.boundingRect(contours[0])
            cx, cy = img_np.shape[1]//2, img_np.shape[0]//2  # centro imagen
            obj_cx, obj_cy = x + w//2, y + h//2             # centro objeto
            dx, dy = cx - obj_cx, cy - obj_cy              # desplazamiento
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            img_centered = cv2.warpAffine(img_np, M, (img_np.shape[1], img_np.shape[0]))
        else:
            img_centered = img_np  # si no hay contornos, dejar igual

        # Convertir a base64
        pil_img = Image.fromarray(img_centered)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG")
        processed_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()

        processed.append({
            "image": processed_b64,
            "vector": cap.get("vector")
        })

    print(f"Procesadas {len(processed)} capturas.")
    return jsonify({"captures": processed})


if __name__ == "__main__":
    app.run(debug=True)
