from flask import Flask, request, jsonify
from flask_cors import CORS
import os, io, base64, requests
from PIL import Image

# === Config ===
REPLICATE_MODEL = "cjwbw/rembg"  # modelo remoto
REPLICATE_API_URL = f"https://api.replicate.com/v1/predictions"
REPLICATE_API_KEY = os.environ.get("REPLICATE_API_KEY")  # ⚠️ la pondrás en Render dashboard

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "✅ Servidor Flask activo y usando RemBG en la nube (Replicate)."

def remove_background_via_replicate(img_base64):
    """Envía la imagen base64 a Replicate y devuelve una versión sin fondo (PNG base64)."""
    try:
        headers = {
            "Authorization": f"Token {REPLICATE_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "version": "a58ed858c6f0e4c1e28e3f6e27315cf816648db6fcbad69f32f3a534e5d9a30e",  # rembg u2netp
            "input": {"image": img_base64}
        }

        resp = requests.post(REPLICATE_API_URL, headers=headers, json=payload)
        if resp.status_code != 201:
            print("Error Replicate:", resp.text)
            return None

        prediction = resp.json()
        status_url = prediction["urls"]["get"]

        # Esperar el resultado (polling)
        for _ in range(30):
            r = requests.get(status_url, headers=headers)
            data = r.json()
            if data["status"] == "succeeded":
                output_url = data["output"][0]
                # Descargar imagen procesada
                img_bytes = requests.get(output_url).content
                img_b64 = "data:image/png;base64," + base64.b64encode(img_bytes).decode("utf-8")
                return img_b64
            elif data["status"] in ["failed", "canceled"]:
                print("Replicate falló:", data)
                return None
        return None
    except Exception as e:
        print("⚠️ Error usando Replicate:", e)
        return None


@app.route("/process", methods=["POST"])
def process_photos():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No se recibió JSON válido"}), 400

    captures = data.get("captures", [])
    if not captures:
        return jsonify({"error": "No se recibieron capturas"}), 400

    processed = []

    for i, cap in enumerate(captures):
        try:
            img_data = cap.get("image")
            if not img_data:
                continue

            # Fondo transparente desde Replicate
            img_trans_b64 = remove_background_via_replicate(img_data)
            if not img_trans_b64:
                img_trans_b64 = img_data  # fallback si falla

            processed.append({
                "image_bg": img_data,  # original
                "image_transparent": img_trans_b64,
                "vector": cap.get("vector", {})
            })

        except Exception as e:
            print(f"❌ Error procesando imagen {i}: {e}")

    print(f"✅ Procesadas {len(processed)} imágenes (via Replicate).")
    return jsonify({"captures": processed})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
