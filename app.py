from flask import Flask, request, jsonify
from flask_cors import CORS
from rembg import remove, new_session
from PIL import Image
import io, base64, os

app = Flask(__name__)
CORS(app)

print("✅ Servidor Flask listo para recibir peticiones.")

# --- Lazy loading del modelo ---
rembg_session = None
def get_rembg_session():
    global rembg_session
    if rembg_session is None:
        print("🔄 Inicializando modelo rembg (u2netp, versión ligera)...")
        rembg_session = new_session("u2netp")  # Modelo pequeño (~12MB)
        print("✅ Modelo rembg listo.")
    return rembg_session

@app.route("/")
def index():
    return "✅ Servidor Flask activo y funcionando en Render."

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
    if not data:
        return jsonify({"error": "No se recibió JSON válido"}), 400

    captures = data.get("captures", [])
    if not captures:
        return jsonify({"error": "No se recibieron capturas"}), 400

    processed = []

    for i, cap in enumerate(captures):
        img_data = cap.get("image")
        if not img_data:
            continue

        try:
            # --- Decodificar base64 ---
            base64_str = img_data.split(",")[1] if "," in img_data else img_data
            img_bytes = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

            # --- Remover fondo con rembg ---
            try:
                session = get_rembg_session()
                img_no_bg = remove(img, session=session)
            except Exception as e:
                print(f"⚠️ Error removiendo fondo en imagen {i}: {e}")
                img_no_bg = img  # fallback sin procesar

            # --- Convertir ambas imágenes a base64 ---
            # Fondo original (JPG)
            buf_bg = io.BytesIO()
            img.convert("RGB").save(buf_bg, format="JPEG", quality=90)
            img_bg_b64 = "data:image/jpeg;base64," + base64.b64encode(buf_bg.getvalue()).decode("utf-8")

            # Fondo transparente (PNG)
            buf_trans = io.BytesIO()
            img_no_bg.save(buf_trans, format="PNG")
            img_trans_b64 = "data:image/png;base64," + base64.b64encode(buf_trans.getvalue()).decode("utf-8")

            processed.append({
                "image_bg": img_bg_b64,
                "image_transparent": img_trans_b64,
                "vector": cap.get("vector", {})
            })

        except Exception as e:
            print(f"❌ Error procesando imagen {i}: {e}")
            continue

    print(f"✅ Procesadas {len(processed)} imágenes correctamente.")
    return jsonify({"captures": processed})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
