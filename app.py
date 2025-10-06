from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import trimesh
from PIL import Image
import os
import uuid

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def detect_shape(img):
    """
    Detecta la forma principal del objeto en el centro.
    Devuelve: 'cylinder', 'box' o 'sphere'
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 'cylinder'

    # Tomar contorno más grande
    c = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    aspect_ratio = w / h

    # Decidir forma aproximada
    if aspect_ratio > 1.2:  # ancho > alto
        return 'box'
    elif aspect_ratio < 0.8:
        return 'cylinder'
    else:
        return 'sphere'

@app.route("/upload", methods=["POST"])
def upload_photos():
    photos = []
    for key in request.files:
        file = request.files[key]
        img = Image.open(file.stream).convert("RGB")
        photos.append(np.array(img))

    if len(photos) == 0:
        return jsonify({"error": "No se enviaron fotos"}), 400

    # Color promedio
    avg_color = np.mean([np.mean(p.reshape(-1,3), axis=0) for p in photos], axis=0)
    color = [c/255.0 for c in avg_color]

    # Detectar forma usando la primera foto (aprox.)
    shape = detect_shape(photos[0])

    # Generar modelo 3D simple
    if shape == 'box':
        mesh = trimesh.creation.box(extents=[1.0, 0.5, 0.5])
    elif shape == 'sphere':
        mesh = trimesh.creation.icosphere(radius=0.5)
    else:  # cylinder
        mesh = trimesh.creation.cylinder(radius=0.5, height=1.0)

    # Colorear
    mesh.visual.vertex_colors = np.tile(np.array(color) * 255, (mesh.vertices.shape[0], 1))

    # Guardar archivo
    filename = f"model_{uuid.uuid4().hex}.obj"
    output_path = os.path.join(UPLOAD_FOLDER, filename)
    mesh.export(output_path)

    return jsonify({"model_url": f"/uploads/{filename}", "photos": len(photos), "shape": shape})

@app.route("/uploads/<filename>")
def serve_model(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/")
def index():
    return "Servidor de reconstrucción 3D simple activo."

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
