from flask import (
    Flask,
    request,
    send_file,
    jsonify
)

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
# CONFIGURATION
# =========================================================

MAX_IMAGE_SIZE = 1000

MAX_FEATURES = 1500

MAX_3D_POINTS = 10000

MIN_MATCHES = 25


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "ok": True,

        "service":
            "ObjectScan 3D",

        "status":
            "running",

        "engine":
            "OpenCV + Open3D",

        "colmap":
            False

    })


# =========================================================
# PROCESS 3D
# =========================================================

@app.route(
    "/process-3d",
    methods=["POST"]
)
def process_3d():

    temp_dir = tempfile.mkdtemp(
        prefix="objectscan_"
    )

    try:

        print("")
        print(
            "=========================================="
        )
        print(
            "       OBJECTSCAN LIGHT 3D"
        )
        print(
            "=========================================="
        )


        # =====================================================
        # 1. RECEIVE IMAGES
        # =====================================================

        files = [

            request.files.get("image1"),

            request.files.get("image2"),

            request.files.get("image3")

        ]


        if any(
            file is None
            for file in files
        ):

            raise Exception(
                "Se requieren image1, image2 e image3."
            )


        image_paths = []


        for index, file in enumerate(
            files
        ):

            path = os.path.join(

                temp_dir,

                f"image{index + 1}.jpg"

            )


            file.save(
                path
            )


            image_paths.append(
                path
            )


        print(
            "Images received."
        )


        # =====================================================
        # 2. LOAD IMAGES
        # =====================================================

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


            images.append(
                image
            )


        print(
            "Image sizes:",
            [
                (
                    image.shape[1],
                    image.shape[0]
                )

                for image in images
            ]
        )


        # =====================================================
        # 3. CAMERA MATRIX
        # =====================================================

        height, width = (
            images[0].shape[:2]
        )


        focal = (
            0.9 *
            max(
                width,
                height
            )
        )


        K = np.array(

            [

                [
                    focal,
                    0,
                    width / 2
                ],

                [
                    0,
                    focal,
                    height / 2
                ],

                [
                    0,
                    0,
                    1
                ]

            ],

            dtype=np.float64

        )


        # =====================================================
        # 4. FEATURES
        # =====================================================

        print(
            "Detecting features..."
        )


        orb = cv2.ORB_create(

            nfeatures=
                MAX_FEATURES,

            scaleFactor=
                1.2,

            nlevels=
                6,

            edgeThreshold=
                15,

            fastThreshold=
                10

        )


        keypoints = []

        descriptors = []


        for image in images:

            gray = cv2.cvtColor(

                image,

                cv2.COLOR_BGR2GRAY

            )


            kp, des = (
                orb.detectAndCompute(
                    gray,
                    None
                )
            )


            if (
                des is None
                or
                len(kp) < 30
            ):

                raise Exception(

                    "No se encontraron "
                    "suficientes características "
                    "en una fotografía."

                )


            keypoints.append(
                kp
            )

            descriptors.append(
                des
            )


        print(
            "Features:",
            [
                len(kp)
                for kp in keypoints
            ]
        )


        # =====================================================
        # 5. MATCHER
        # =====================================================

        matcher = cv2.BFMatcher(

            cv2.NORM_HAMMING,

            crossCheck=False

        )


        # =====================================================
        # 6. MATCH 1 → 2
        # =====================================================

        pts1, pts2 = match_points(

            keypoints[0],

            descriptors[0],

            keypoints[1],

            descriptors[1],

            matcher

        )


        print(
            "Good matches 1-2:",
            len(pts1)
        )


        if len(pts1) < MIN_MATCHES:

            raise Exception(

                "Las fotografías 1 y 2 no "
                "tienen suficientes puntos "
                "en común. Usa más textura "
                "o mueve el teléfono menos "
                "entre fotografías."

            )


        # =====================================================
        # 7. ESSENTIAL MATRIX
        # =====================================================

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

                "No se pudo calcular la "
                "geometría entre las fotografías."

            )


        # =====================================================
        # 8. RECOVER CAMERA POSE
        # =====================================================

        _, R, t, pose_mask = (
            cv2.recoverPose(

                E,

                pts1,

                pts2,

                K

            )
        )


        valid = (
            pose_mask.ravel() > 0
        )


        pts1 = pts1[valid]

        pts2 = pts2[valid]


        print(
            "Pose inliers:",
            len(pts1)
        )


        if len(pts1) < 15:

            raise Exception(

                "La geometría de cámara "
                "no es suficientemente estable."

            )


        # =====================================================
        # 9. TRIANGULATION
        # =====================================================

        P1 = (

            K @

            np.hstack(

                (

                    np.eye(3),

                    np.zeros(
                        (3, 1)
                    )

                )

            )

        )


        P2 = (

            K @

            np.hstack(

                (

                    R,

                    t

                )

            )

        )


        points_4d = (
            cv2.triangulatePoints(

                P1,

                P2,

                pts1.T,

                pts2.T

            )
        )


        w_values = (
            points_4d[3]
        )


        valid_w = (
            np.abs(w_values)
            > 1e-8
        )


        points_4d = (
            points_4d[:, valid_w]
        )


        points_3d = (

            points_4d[:3]

            /

            points_4d[3]

        ).T


        print(
            "Raw 3D points:",
            len(points_3d)
        )


        # =====================================================
        # 10. FILTER
        # =====================================================

        points_3d = filter_points(
            points_3d
        )


        print(
            "Filtered 3D points:",
            len(points_3d)
        )


        if len(points_3d) < 30:

            raise Exception(

                "La triangulación produjo "
                "muy pocos puntos 3D."

            )


        # =====================================================
        # 11. THIRD IMAGE VALIDATION
        # =====================================================

        validate_third_image(

            images[2],

            keypoints[2],

            descriptors[2],

            matcher

        )


        # =====================================================
        # 12. LIMIT POINTS
        # =====================================================

        if (
            len(points_3d)
            >
            MAX_3D_POINTS
        ):

            indices = np.random.choice(

                len(points_3d),

                MAX_3D_POINTS,

                replace=False

            )


            points_3d = (
                points_3d[indices]
            )


        print(
            "Final points:",
            len(points_3d)
        )


        # =====================================================
        # 13. NORMALIZE
        # =====================================================

        points_3d = normalize_points(
            points_3d
        )


        # =====================================================
        # 14. OPEN3D POINT CLOUD
        # =====================================================

        pcd = (
            o3d.geometry.PointCloud()
        )


        pcd.points = (
            o3d.utility
            .Vector3dVector(
                points_3d
            )
        )


        print(
            "Open3D points:",
            len(pcd.points)
        )


        # =====================================================
        # 15. VOXEL
        # =====================================================

        pcd = (
            pcd.voxel_down_sample(
                voxel_size=0.025
            )
        )


        print(
            "After voxel:",
            len(pcd.points)
        )


        if len(pcd.points) < 30:

            raise Exception(

                "La nube de puntos "
                "es demasiado pequeña."

            )


        # =====================================================
        # 16. REMOVE OUTLIERS
        # =====================================================

        try:

            pcd, _ = (
                pcd.remove_statistical_outlier(

                    nb_neighbors=8,

                    std_ratio=2.0

                )
            )

        except Exception as error:

            print(
                "Outlier removal skipped:",
                error
            )


        # =====================================================
        # 17. NORMALS
        # =====================================================

        print(
            "Estimating normals..."
        )


        pcd.estimate_normals(

            search_param=

            o3d.geometry
            .KDTreeSearchParamHybrid(

                radius=0.08,

                max_nn=20

            )

        )


        # =====================================================
        # 18. BALL PIVOTING
        # =====================================================

        print(
            "Creating surface..."
        )


        distances = (
            pcd
            .compute_nearest_neighbor_distance()
        )


        if len(distances) == 0:

            raise Exception(
                "No se pudieron calcular "
                "las distancias de vecinos."
            )


        median_distance = float(

            np.median(
                distances
            )

        )


        if (
            not np.isfinite(
                median_distance
            )
            or
            median_distance <= 0
        ):

            median_distance = 0.02


        radii = (
            o3d.utility
            .DoubleVector(

                [

                    median_distance * 1.5,

                    median_distance * 2.5,

                    median_distance * 4.0

                ]

            )
        )


        mesh = (

            o3d.geometry
            .TriangleMesh
            .create_from_point_cloud_ball_pivoting(

                pcd,

                radii

            )

        )


        if (
            len(mesh.vertices)
            == 0
        ):

            raise Exception(

                "No se pudo crear una "
                "superficie 3D."

            )


        print(
            "Mesh vertices:",
            len(mesh.vertices)
        )


        print(
            "Mesh triangles:",
            len(mesh.triangles)
        )


        # =====================================================
        # 19. CLEAN MESH
        # =====================================================

        mesh.remove_degenerate_triangles()

        mesh.remove_duplicated_triangles()

        mesh.remove_duplicated_vertices()

        mesh.remove_non_manifold_edges()

        mesh.compute_vertex_normals()


        # =====================================================
        # 20. DECIMATE
        # =====================================================

        if (
            len(mesh.triangles)
            > 12000
        ):

            print(
                "Reducing mesh..."
            )


            try:

                mesh =
                    mesh.simplify_quadric_decimation(
                        12000
                    )

                mesh.compute_vertex_normals()

            except Exception as error:

                print(
                    "Decimation skipped:",
                    error
                )


        # =====================================================
        # 21. SAVE PLY
        # =====================================================

        ply_path = os.path.join(

            temp_dir,

            "model.ply"

        )


        glb_path = os.path.join(

            temp_dir,

            "objectscan.glb"

        )


        success = (
            o3d.io.write_triangle_mesh(

                ply_path,

                mesh

            )
        )


        if not success:

            raise Exception(
                "No se pudo guardar model.ply."
            )


        # =====================================================
        # 22. GLB
        # =====================================================

        convert_to_glb(

            ply_path,

            glb_path

        )


        if not os.path.exists(
            glb_path
        ):

            raise Exception(
                "No se generó objectscan.glb."
            )


        size = os.path.getsize(
            glb_path
        )


        print(
            "GLB size:",
            size,
            "bytes"
        )


        print(
            "=========================================="
        )

        print(
            "OBJECTSCAN SUCCESS"
        )

        print(
            "=========================================="
        )


        # =====================================================
        # 23. RETURN
        # =====================================================

        return send_file(

            glb_path,

            mimetype=
                "model/gltf-binary",

            as_attachment=False,

            download_name=
                "objectscan.glb"

        )


    except Exception as error:

        print("")
        print(
            "=========================================="
        )

        print(
            "OBJECTSCAN ERROR"
        )

        print(
            "=========================================="
        )


        traceback.print_exc()


        return jsonify({

            "ok": False,

            "error":
                str(error)

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

    height, width = (
        image.shape[:2]
    )


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


    new_width = max(

        1,

        int(
            width * scale
        )

    )


    new_height = max(

        1,

        int(
            height * scale
        )

    )


    return cv2.resize(

        image,

        (
            new_width,
            new_height
        ),

        interpolation=
            cv2.INTER_AREA

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


        if (
            m.distance
            <
            0.72 * n.distance
        ):

            good.append(
                m
            )


    if len(good) > MAX_FEATURES:

        good = sorted(

            good,

            key=lambda x:
                x.distance

        )[
            :MAX_FEATURES
        ]


    pts1 = np.float64([

        kp1[
            match.queryIdx
        ].pt

        for match in good

    ])


    pts2 = np.float64([

        kp2[
            match.trainIdx
        ].pt

        for match in good

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


    if len(points) == 0:

        return points


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


    # ---------------------------------------------
    # REMOVE EXTREME DEPTH
    # ---------------------------------------------

    z = points[:, 2]


    positive = (
        z > 0
    )


    points = points[
        positive
    ]


    if len(points) == 0:

        return points


    z = points[:, 2]


    low = np.percentile(
        z,
        3
    )


    high = np.percentile(
        z,
        97
    )


    mask = (

        (z >= low)

        &

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


    points = (
        points - center
    )


    distances = np.linalg.norm(

        points,

        axis=1

    )


    scale = np.percentile(

        distances,

        90

    )


    if (
        not np.isfinite(scale)
        or
        scale <= 1e-9
    ):

        scale = 1.0


    points = (
        points / scale
    )


    distances = np.linalg.norm(

        points,

        axis=1

    )


    points = points[
        distances < 3.0
    ]


    return points


# =========================================================
# THIRD IMAGE
# =========================================================

def validate_third_image(

    image,

    keypoints,

    descriptors,

    matcher

):

    print(
        "Validating third image..."
    )


    gray = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2GRAY

    )


    corners = cv2.goodFeaturesToTrack(

        gray,

        maxCorners=300,

        qualityLevel=0.02,

        minDistance=8

    )


    if corners is not None:

        print(

            "Third image corners:",

            len(corners)

        )


    if (
        descriptors is not None
        and
        len(descriptors) > 0
    ):

        print(
            "Third image ORB:",
            len(keypoints)
        )


# =========================================================
# GLB
# =========================================================

def convert_to_glb(

    mesh_path,

    glb_path

):

    print(
        "Converting to GLB..."
    )


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
