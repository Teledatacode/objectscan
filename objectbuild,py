import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
import open3d as o3d
from datetime import datetime

app = Flask(__name__)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload():
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(folder, exist_ok=True)

    files = request.files
    filenames = []
    for key in files:
        file = files[key]
        path = os.path.join(folder, file.filename)
        file.save(path)
        filenames.append(path)

    if len(filenames) < 2:
        return jsonify({"error": "Se necesitan al menos 2 fotos"}), 400

    # Procesar (reconstrucción 3D básica)
    model_path = os.path.join(OUTPUT_DIR, f"model_{job_id}.ply")
    build_lowpoly_model(filenames, model_path)

    return jsonify({
        "status": "ok",
        "model": f"/download/{os.path.basename(model_path)}"
    })


@app.route("/download/<filename>")
def download(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "Archivo no encontrado"}), 404
    return send_file(path, as_attachment=True)


def build_lowpoly_model(image_paths, output_path):
    """
    Crea un modelo 3D simple usando correspondencias de características
    (esto NO es fotogrametría completa, solo un ejemplo base).
    """
    # Leer imágenes en escala de grises
    images = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in image_paths]
    sift = cv2.SIFT_create()

    # Detectar keypoints y descriptores
    kp1, des1 = sift.detectAndCompute(images[0], None)
    kp2, des2 = sift.detectAndCompute(images[1], None)

    # Emparejar características
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    # Calcular matriz esencial y recuperar pose
    K = np.array([[1000, 0, 640],
                  [0, 1000, 360],
                  [0, 0, 1]])
    E, mask = cv2.findEssentialMat(pts1, pts2, K)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    # Triangulación básica
    proj1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    proj2 = np.hstack((R, t))
    pts4d = cv2.triangulatePoints(K @ proj1, K @ proj2,
                                  pts1.T, pts2.T)
    pts3d = (pts4d[:3] / pts4d[3]).T

    # Crear nube de puntos Open3D
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts3d)
    cloud = cloud.voxel_down_sample(voxel_size=0.01)
    o3d.io.write_point_cloud(output_path, cloud)
    print(f"✅ Modelo guardado en: {output_path}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
