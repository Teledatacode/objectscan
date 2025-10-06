from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import trimesh
import os
from PIL import Image
import uuid

app = Flask(__name__)
CORS(app)  # Permite llamadas desde cualquier origen

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_photos():
    photos = []
    for key in request.files:
        file = request.files[key]
        img = Image.open(file.stream).convert("RGB")
        photos.append(np.array(img))
    
    if len(photos) == 0:
        return jsonify({"error": "No se enviaron fotos"}), 400
    
    # Promedio de color
    avg_color = np.mean([np.mean(p.reshape(-1, 3), axis=0) for p in photos], axis=0)
    color = [c / 255.0 for c in avg_color]

    # Crear un modelo simple: cilindro + tapa
    height = 1.0
    radius = 0.5
    mesh_cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
    mesh_cap = trimesh.creation.cylinder(radius=radius * 1.3, height=0.1, sections=32)
    mesh_cap.apply_translation([0, 0, -height / 2 - 0.05])

    combined = trimesh.util.concatenate([mesh_cylinder, mesh_cap])
    combined.visual.vertex_colors = np.tile(np.array(color) * 255, (combined.vertices.shape[0], 1))

    # Nombre único para evitar sobrescrituras
    filename = f"model_{uuid.uuid4().hex}.obj"
    output_path = os.path.join(UPLOAD_FOLDER, filename)
    combined.export(output_path)

    # Devuelve JSON con la URL del modelo
    return jsonify({"model_url": f"/uploads/{filename}", "photos": len(photos)})

# Servir archivos OBJ desde /uploads
@app.route("/uploads/<filename>")
def serve_model(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/")
def index():
    return "Servidor de reconstrucción 3D simplificada activo."

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
