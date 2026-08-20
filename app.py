from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

import os
import gc
import shutil
import tempfile
import traceback

import cv2
import numpy as np
import open3d as o3d
import trimesh


app = Flask(__name__)
CORS(app)


# =========================================================
# CONFIGURACIÓN PARA 512 MB
# =========================================================

MAX_IMAGE_SIZE = 900
MAX_FEATURES = 1200
MAX_3D_POINTS = 12000

# Separación virtual de cámaras.
# La reconstrucción será a escala arbitraria.
BASELINE = 1.0


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "ok": True,
        "service": "ObjectScan 3D",
        "status": "running",
        "engine": "OpenCV + Open3D",
        "colmap": False
    })


# =========================================================
# PROCESS 3D
# =========================================================

@app.route("/process-3d", methods=["POST"])
def process_3d():

    temp_dir = tempfile.mkdtemp(
        prefix="objectscan_"
    )

    try:

        print("\n================================")
        print("OBJECTSCAN LIGHT 3D")
        print("================================")

        # -------------------------------------------------
        # 1. RECIBIR IMÁGENES
        # -------------------------------------------------

        files = [
            request.files.get("image1"),
            request.files.get("image2"),
            request.files.get("image3")
        ]

        if any(f is None for f in files):

            raise Exception(
                "Se requieren image1, image2 e image3."
            )

        image_paths = []

        for i, file in enumerate(files):

            path = os.path.join(
                temp_dir,
                f"image{i + 1}.jpg"
            )

            file.save(path)

            image_paths.append(path)

        print("Images saved.")

        # -------------------------------------------------
        # 2. CARGAR Y REDUCIR
        # -------------------------------------------------

        images = []

        for path in image_paths:

            image = cv2.imread(
                path,
                cv2.IMREAD_COLOR
            )

            if image is None:

                raise Exception(
                    f"No se pudo leer {path}"
                )

            image = resize_image(
                image,
                MAX_IMAGE_SIZE
            )

            images.append(image)

        print(
            "Images loaded:",
            [
                (img.shape[1], img.shape[0])
                for img in images
            ]
        )

        # -------------------------------------------------
        # 3. INTRÍNSECOS APROXIMADOS
        # -------------------------------------------------

        h, w = images[0].shape[:2]

        focal = 0.9 * max(
            w,
            h
        )

        K = np.array(
            [
                [focal, 0, w / 2],
                [0, focal, h / 2],
                [0, 0, 1]
            ],
            dtype=np.float64
        )

        # -------------------------------------------------
        # 4. FEATURES
        # -------------------------------------------------

        orb = cv2.ORB_create(
            nfeatures=MAX_FEATURES,
            scaleFactor=1.2,
            nlevels=5,
            fastThreshold=15
        )

        keypoints = []
        descriptors = []

        for image in images:

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            kp, des = orb.detectAndCompute(
                gray,
                None
            )

            if des is None or len(kp) < 30:

                raise Exception(
                    "No se encontraron suficientes "
                    "características en una de las fotos."
                )

            keypoints.append(kp)
            descriptors.append(des)

        print(
            "Features:",
            [len(k) for k in keypoints]
        )

        # -------------------------------------------------
        # 5. MATCHER
        # -------------------------------------------------

        matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=False
        )

        # -------------------------------------------------
        # 6. RECONSTRUIR 1 → 2
        # -------------------------------------------------

        pts1, pts2 = match_points(
            keypoints[0],
            descriptors[0],
            keypoints[1],
            descriptors[1],
            matcher
        )

        if len(pts1) < 20:

            raise Exception(
                "No hay suficientes puntos "
                "comunes entre las fotos 1 y 2."
            )

        print(
            "Matches 1-2:",
            len(pts1)
        )

        E, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.5
        )

        if E is None:

            raise Exception(
                "No se pudo calcular la geometría "
                "entre las fotos 1 y 2."
            )

        _, R, t, pose_mask = cv2.recoverPose(
            E,
            pts1,
            pts2,
            K
        )

        valid = (
            pose_mask.ravel() > 0
        )

        pts1 = pts1[valid]
        pts2 = pts2[valid]

        if len(pts1) < 15:

            raise Exception(
                "La geometría entre las fotos "
                "no es suficientemente estable."
            )

        # -------------------------------------------------
        # 7. TRIANGULACIÓN
        # -------------------------------------------------

        P1 = K @ np.hstack(
            (
                np.eye(3),
                np.zeros(
                    (3, 1)
                )
            )
        )

        P2 = K @ np.hstack(
            (
                R,
                t * BASELINE
            )
        )

        points_4d = cv2.triangulatePoints(
            P1,
            P2,
            pts1.T,
            pts2.T
        )

        points_3d = (
            points_4d[:3] /
            points_4d[3]
        ).T

        # -------------------------------------------------
        # 8. FILTRAR PUNTOS
        # -------------------------------------------------

        points_3d = filter_points(
            points_3d
        )

        print(
            "3D points after filtering:",
            len(points_3d)
        )

        # -------------------------------------------------
        # 9. AGREGAR LA TERCERA FOTO
        # -------------------------------------------------

        try:

            points_3d = add_third_view(
                points_3d,
                keypoints[2],
                descriptors[2],
                images,
                K
            )

        except Exception as error:

            print(
                "Third-view refinement skipped:",
                error
            )

        # -------------------------------------------------
        # 10. LIMITAR MEMORIA
        # -------------------------------------------------

        if len(points_3d) > MAX_3D_POINTS:

            indices = np.random.choice(
                len(points_3d),
                MAX_3D_POINTS,
                replace=False
            )

            points_3d = points_3d[
                indices
            ]

        print(
            "Final points:",
            len(points_3d)
        )

        if len(points_3d) < 30:

            raise Exception(
                "La reconstrucción produjo "
                "muy pocos puntos 3D."
            )

        # -------------------------------------------------
        # 11. NORMALIZAR
        # -------------------------------------------------

        points_3d = normalize_points(
            points_3d
        )

        # -------------------------------------------------
        # 12. POINT CLOUD
        # -------------------------------------------------

        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(
            points_3d
        )

        # -------------------------------------------------
        # 13. VOXEL DOWN SAMPLE
        # -------------------------------------------------

        pcd = pcd.voxel_down_sample(
            voxel_size=0.025
        )

        print(
            "Point cloud:",
            len(pcd.points)
        )

        if len(pcd.points) < 30:

            raise Exception(
                "La nube de puntos es demasiado pequeña."
            )

        # -------------------------------------------------
        # 14. LIMPIEZA
        # -------------------------------------------------

        pcd, _ = (
            pcd.remove_statistical_outlier(
                nb_neighbors=10,
                std_ratio=2.0
            )
        )

        # -------------------------------------------------
        # 15. NORMALES
        # -------------------------------------------------

        pcd.estimate_normals(
            search_param=
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.08,
                max_nn=20
            )
        )

        pcd.orient_normals_consistent_tangent_plane(
            10
        )

        # -------------------------------------------------
        # 16. BALL PIVOTING
        # -------------------------------------------------

        distances = pcd.compute_nearest_neighbor_distance()

        if len(distances) == 0:

            raise Exception(
                "No se pudieron calcular distancias."
            )

        median_distance = float(
            np.median(distances)
        )

        if median_distance <= 0:

            median_distance = 0.02

        radii = o3d.utility.DoubleVector(
            [
                median_distance * 1.5,
                median_distance * 2.5,
                median_distance * 4.0
            ]
        )

        print(
            "Ball Pivoting..."
        )

        mesh = (
            o3d.geometry
            .TriangleMesh
            .create_from_point_cloud_ball_pivoting(
                pcd,
                radii
            )
        )

        if len(mesh.vertices) == 0:

            raise Exception(
                "Ball Pivoting no pudo crear una superficie."
            )

        # -------------------------------------------------
        # 17. LIMPIEZA MESH
        # -------------------------------------------------

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        mesh.compute_vertex_normals()

        # -------------------------------------------------
        # 18. DECIMATE
        # -------------------------------------------------

        if len(mesh.triangles) > 20000:

            mesh = mesh.simplify_quadric_decimation(
                20000
            )

            mesh.compute_vertex_normals()

        # -------------------------------------------------
        # 19. EXPORTAR GLB
        # -------------------------------------------------

        ply_path = os.path.join(
            temp_dir,
            "model.ply"
        )

        glb_path = os.path.join(
            temp_dir,
            "objectscan.glb"
        )

        success = o3d.io.write_triangle_mesh(
            ply_path,
            mesh
        )

        if not success:

            raise Exception(
                "No se pudo guardar la malla."
            )

        convert_to_glb(
            ply_path,
            glb_path
        )

        if not os.path.exists(glb_path):

            raise Exception(
                "No se generó el GLB."
            )

        print(
            "GLB:",
            os.path.getsize(glb_path),
            "bytes"
        )

        return send_file(
            glb_path,
            mimetype="model/gltf-binary",
            as_attachment=False,
            download_name="objectscan.glb"
        )

    except Exception as error:

        print("\n================================")
        print("OBJECTSCAN ERROR")
        print("================================")

        traceback.print_exc()

        return jsonify({
            "ok": False,
            "error": str(error)
        }), 500

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

        gc.collect()


# =========================================================
# RESIZE
# =========================================================

def resize_image(
    image,
    max_size
):

    height, width = image.shape[:2]

    largest = max(
        width,
        height
    )

    if largest <= max_size:

        return image

    scale = (
        max_size /
        float(largest)
    )

    new_width = int(
        width * scale
    )

    new_height = int(
        height * scale
    )

    return cv2.resize(
        image,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )


# =========================================================
# MATCH POINTS
# =========================================================

def match_points(
    kp1,
    des1,
    kp2,
    des2,
    matcher
):

    raw = matcher.knnMatch(
        des1,
        des2,
        k=2
    )

    good = []

    for pair in raw:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.72 * n.distance:

            good.append(m)

    if len(good) > MAX_FEATURES:

        good = sorted(
            good,
            key=lambda x: x.distance
        )[:MAX_FEATURES]

    pts1 = np.float64([
        kp1[m.queryIdx].pt
        for m in good
    ])

    pts2 = np.float64([
        kp2[m.trainIdx].pt
        for m in good
    ])

    return pts1, pts2


# =========================================================
# FILTER POINTS
# =========================================================

def filter_points(
    points
):

    points = np.asarray(
        points,
        dtype=np.float64
    )

    valid = np.isfinite(
        points
    ).all(
        axis=1
    )

    points = points[
        valid
    ]

    if len(points) == 0:

        return points

    # Evitar puntos detrás de la cámara
    points = points[
        points[:, 2] > 0
    ]

    if len(points) == 0:

        return points

    # Eliminar valores extremos
    z = points[:, 2]

    low = np.percentile(
        z,
        2
    )

    high = np.percentile(
        z,
        98
    )

    mask = (
        (z >= low) &
        (z <= high)
    )

    points = points[
        mask
    ]

    return points


# =========================================================
# NORMALIZE
# =========================================================

def normalize_points(
    points
):

    points = np.asarray(
        points,
        dtype=np.float64
    )

    center = np.median(
        points,
        axis=0
    )

    points = points - center

    scale = np.percentile(
        np.linalg.norm(
            points,
            axis=1
        ),
        90
    )

    if scale <= 1e-9:

        scale = 1.0

    points /= scale

    # Recorte final para evitar
    # outliers extremos
    distance = np.linalg.norm(
        points,
        axis=1
    )

    points = points[
        distance < 3.0
    ]

    return points


# =========================================================
# TERCERA VISTA
# =========================================================

def add_third_view(
    points_3d,
    kp3,
    des3,
    images,
    K
):

    # Esta versión usa la tercera imagen
    # como validación ligera.
    #
    # No intenta hacer una reconstrucción
    # densa porque eso consumiría demasiada RAM.

    gray3 = cv2.cvtColor(
        images[2],
        cv2.COLOR_BGR2GRAY
    )

    # Detectar esquinas adicionales
    corners = cv2.goodFeaturesToTrack(
        gray3,
        maxCorners=400,
        qualityLevel=0.02,
        minDistance=8
    )

    if corners is None:

        return points_3d

    # No generamos una nube artificial.
    # La tercera imagen se utiliza principalmente
    # para comprobar que existe contenido visual.

    print(
        "Third image features:",
        len(corners)
    )

    return points_3d


# =========================================================
# OPEN3D → GLB
# =========================================================

def convert_to_glb(
    mesh_path,
    glb_path
):

    mesh = trimesh.load(
        mesh_path,
        process=False
    )

    if isinstance(
        mesh,
        trimesh.Scene
    ):

        scene = mesh

    else:

        scene = trimesh.Scene(
            mesh
        )

    scene.export(
        glb_path,
        file_type="glb"
    )


# =========================================================
# SERVER
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
