from flask import Flask, request, jsonify
from flask_cors import CORS
from rembg import remove
from PIL import Image
import io, base64, os
import numpy as np

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/process", methods=["POST"])
def process_photos():
    """
    Recibe capturas del cliente:
    [
      { "image": "data:image/jpeg;base64,...", "vector": {...} }
    ]
    Devuelve las mismas fotos centradas y con una versión con fondo transparente (PNG).
    """

    data = request.get_json()
    captures = data.get("captures", [])

    processed = []
    for i, cap in enumerate(captures):
        img_data = cap.get("image")
        if not img_data:
            continue

        # --- Decodificar base64 ---
        base64_str = img_data.split(",")[1] if "," in img_data else img_data
        img_bytes = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

        # --- Remover fondo ---
        try:
            img_no_bg = remove(img)
        except Exception as e:
            print(f"Error removiendo fondo en imagen {i}: {e}")
            img_no_bg = img  # fallback sin procesar

        # --- Convertir ambas imágenes a base64 ---
        # Fondo original (convertido a JPG)
        buf_bg = io.BytesIO()
        img.convert("RGB").save(buf_bg, format="JPEG", quality=90)
        img_bg_b64 = "data:image/jpeg;base64," + base64.b64encode(buf_bg.getvalue()).decode("utf-8")

        # Fondo transparente (en PNG)
        buf_trans = io.BytesIO()
        img_no_bg.save(buf_trans, format="PNG")
        img_trans_b64 = "data:image/png;base64," + base64.b64encode(buf_trans.getvalue()).decode("utf-8")

        processed.append({
            "image_bg": img_bg_b64,
            "image_transparent": img_trans_b64,
            "vector": cap.get("vector", {})
        })

    print(f"Procesadas {len(processed)} imágenes correctamente.")
    return jsonify({"captures": processed})
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
