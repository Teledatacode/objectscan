from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import numpy as np

app = Flask(__name__)
CORS(app)  # Permite peticiones desde cualquier origen

@app.route("/process", methods=["POST"])
def process_images():
    try:
        data = request.json
        captures = data.get("captures", [])
        print("Recibidas capturas:", len(captures))

        processed = []

        for idx, cap in enumerate(captures):
            img_str = cap.get("image")
            if not img_str:
                print(f"Captura {idx} vacía")
                continue

            # Asegurarse de que es un base64 completo
            if "," in img_str:
                img_b64 = img_str.split(",")[1]
            else:
                img_b64 = img_str

            img_bytes = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = np.array(img)

            # --- Centrar el objeto: placeholder recorte central ---
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
        print("Error procesando imágenes:", e)
        return jsonify({"captures": [], "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
