from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import trimesh
import os
from PIL import Image
from io import BytesIO

app = Flask(__name__)

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

    # Crear una forma simple (cilindro como sombrero)
    height = 1.0
    radius = 0.5
    mesh_cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
    mesh_cap = trimesh.creation.cylinder(radius=radius * 1.3, height=0.1, sections=32)
    mesh_cap.apply_translation([0, 0, -height / 2 - 0.05])

    combined = trimesh.util.concatenate([mesh_cylinder, mesh_cap])
    combined.visual.vertex_colors = np.tile(np.array(color) * 255, (combined.vertices.shape[0], 1))

    output_path = os.path.join(UPLOAD_FOLDER, "model.obj")
    combined.export(output_path)

    return send_file(output_path, as_attachment=True)

@app.route("/")
def index():
    return "Servidor de reconstrucción 3D simplificada activo."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
