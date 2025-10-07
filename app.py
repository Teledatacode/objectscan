from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)  # permite llamadas desde cualquier origen

@app.route("/process", methods=["POST"])
def process_images():
    data = request.json
    captures = data.get("captures", [])

    processed = []

    for idx, cap in enumerate(captures):
        img_b64 = cap.get("image")
        if not img_b64:
            print(f"Captura {idx} vacía")
            continue

        # --- Convertir base64 a numpy array ---
        img_bytes = base64.b64decode(img_b64.split(",")[-1])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)

        # --- Centrar el objeto usando centro de masa ---
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)

        if coords is not None:
            M = coords.mean(axis=0)[0]  # centro de masa (x, y)
            cx, cy = img_np.shape[1]//2, img_np.shape[0]//2
            dx, dy = cx - M[0], cy - M[1]
            M_translate = np.float32([[1, 0, dx], [0, 1, dy]])
            img_centered = cv2.warpAffine(img_np, M_translate, (img_np.shape[1], img_np.shape[0]))
        else:
            img_centered = img_np

        # --- Convertir de vuelta a base64 ---
        pil_img = Image.fromarray(img_centered)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG")
        processed_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()

        processed.append({
            "image": processed_b64,
            "vector": cap.get("vector")
        })

    print(f"Recibidas capturas: {len(captures)}, procesadas: {len(processed)}")
    return jsonify({"captures": processed})


if __name__ == "__main__":
    app.run(debug=True)
