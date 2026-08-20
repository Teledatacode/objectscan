from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

import os
import shutil
import subprocess
import tempfile
import traceback

import numpy as np
import open3d as o3d
import trimesh


app = Flask(__name__)

CORS(app)


# =========================================================
# CONFIG
# =========================================================

PORT = int(os.environ.get("PORT", 10000))

DEPTH_ENABLED = os.environ.get(
    "DEPTH_ENABLED",
    "false"
).lower() == "true"

DEPTH_MODEL = os.environ.get(
    "DEPTH_MODEL",
    "LiheYoung/depth-anything-small-hf"
)


# =========================================================
# DEPTH MODEL
# =========================================================

depth_pipeline = None


def load_depth_model():

    global depth_pipeline

    if not DEPTH_ENABLED:
        print("Depth Anything disabled.")
        return None

    if depth_pipeline is not None:
        return depth_pipeline

    print("")
    print("==========================================")
    print("LOADING DEPTH ANYTHING")
    print("==========================================")

    try:

        from transformers import pipeline

        depth_pipeline = pipeline(
            "depth-estimation",
            model=DEPTH_MODEL
        )

        print(
            "Depth model loaded:",
            DEPTH_MODEL
        )

        return depth_pipeline

    except Exception as error:

        print(
            "Could not load Depth Anything."
        )

        print(error)

        traceback.print_exc()

        depth_pipeline = None

        return None


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "ok": True,
        "service": "ObjectScan 3D",
        "status": "running",
        "depth_enabled": DEPTH_ENABLED,
        "depth_model": DEPTH_MODEL
    })


@app.route("/health", methods=["GET"])
def health():

    colmap_ok = False
    colmap_error = None

    try:

        result = subprocess.run(
            ["colmap", "-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10
        )

        colmap_ok = (
            result.returncode == 0
        )

    except Exception as error:

        colmap_error = str(error)


    return jsonify({

        "ok":
            colmap_ok,

        "colmap":
            colmap_ok,

        "colmap_error":
            colmap_error,

        "depth_enabled":
            DEPTH_ENABLED,

        "depth_model":
            DEPTH_MODEL

    }), 200 if colmap_ok else 500


# =========================================================
# DEPTH ESTIMATION
# =========================================================

def run_depth_estimation(
    image_path,
    output_path
):

    model = load_depth_model()

    if model is None:

        return False


    print("")
    print(
        "Running depth estimation:",
        image_path
    )


    try:

        from PIL import Image

        image = Image.open(
            image_path
        ).convert("RGB")


        result = model(
            image
        )


        depth = result["predicted_depth"]


        if hasattr(
            depth,
            "detach"
        ):

            depth = (
                depth
                .detach()
                .cpu()
                .numpy()
            )


        depth = np.asarray(
            depth
        )


        depth = np.squeeze(
            depth
        )


        if depth.ndim != 2:

            raise Exception(
                "Invalid depth output."
            )


        # -----------------------------------------------------
        # NORMALIZE
        # -----------------------------------------------------

        minimum = float(
            depth.min()
        )

        maximum = float(
            depth.max()
        )


        if maximum <= minimum:

            raise Exception(
                "Depth map has no useful range."
            )


        normalized = (
            (depth - minimum)
            /
            (maximum - minimum)
        )


        depth_image = (
            normalized * 255.0
        ).astype(
            np.uint8
        )


        from PIL import Image

        Image.fromarray(
            depth_image
        ).save(
            output_path
        )


        print(
            "Depth saved:",
            output_path
        )


        return True


    except Exception as error:

        print(
            "Depth estimation failed:"
        )

        print(error)

        traceback.print_exc()

        return False


# =========================================================
# PROCESS 3D
# =========================================================

@app.route(
    "/process-3d",
    methods=["POST"]
)
def process_3d():

    print("")
    print("==========================================")
    print("        OBJECTSCAN 3D REQUEST")
    print("==========================================")


    temp_dir = tempfile.mkdtemp(
        prefix="objectscan_"
    )


    try:

        # =====================================================
        # CHECK COLMAP
        # =====================================================

        check_colmap()


        # =====================================================
        # RECEIVE IMAGES
        # =====================================================

        image1 = request.files.get(
            "image1"
        )

        image2 = request.files.get(
            "image2"
        )

        image3 = request.files.get(
            "image3"
        )


        if not image1:
            raise Exception(
                "Missing image1"
            )


        if not image2:
            raise Exception(
                "Missing image2"
            )


        if not image3:
            raise Exception(
                "Missing image3"
            )


        # =====================================================
        # IMAGE DIRECTORY
        # =====================================================

        images_dir = os.path.join(
            temp_dir,
            "images"
        )


        os.makedirs(
            images_dir,
            exist_ok=True
        )


        image_paths = [

            os.path.join(
                images_dir,
                "image1.jpg"
            ),

            os.path.join(
                images_dir,
                "image2.jpg"
            ),

            os.path.join(
                images_dir,
                "image3.jpg"
            )

        ]


        image1.save(
            image_paths[0]
        )

        image2.save(
            image_paths[1]
        )

        image3.save(
            image_paths[2]
        )


        print(
            "Images received successfully."
        )


        # =====================================================
        # VERIFY IMAGES
        # =====================================================

        for path in image_paths:

            if not os.path.exists(
                path
            ):

                raise Exception(
                    f"Image was not saved: {path}"
                )


            print(
                path,
                os.path.getsize(path),
                "bytes"
            )


        # =====================================================
        # DEPTH MAPS
        # =====================================================

        depth_dir = os.path.join(
            temp_dir,
            "depth"
        )


        os.makedirs(
            depth_dir,
            exist_ok=True
        )


        depth_paths = []


        if DEPTH_ENABLED:

            print("")
            print(
                "Generating depth maps..."
            )


            for index, image_path in enumerate(
                image_paths,
                start=1
            ):

                depth_path = os.path.join(
                    depth_dir,
                    f"depth{index}.png"
                )


                success = run_depth_estimation(
                    image_path,
                    depth_path
                )


                if success:

                    depth_paths.append(
                        depth_path
                    )


        print(
            "Depth maps generated:",
            len(depth_paths)
        )


        # =====================================================
        # COLMAP DIRECTORIES
        # =====================================================

        database_path = os.path.join(
            temp_dir,
            "database.db"
        )


        sparse_dir = os.path.join(
            temp_dir,
            "sparse"
        )


        dense_dir = os.path.join(
            temp_dir,
            "dense"
        )


        os.makedirs(
            sparse_dir,
            exist_ok=True
        )


        os.makedirs(
            dense_dir,
            exist_ok=True
        )


        # =====================================================
        # FEATURE EXTRACTION
        # =====================================================

        run_command([

            "colmap",

            "feature_extractor",

            "--database_path",
            database_path,

            "--image_path",
            images_dir,

            "--ImageReader.single_camera",
            "1",

            "--ImageReader.camera_model",
            "SIMPLE_RADIAL",

            "--SiftExtraction.use_gpu",
            "0"

        ])


        # =====================================================
        # FEATURE MATCHING
        # =====================================================

        run_command([

            "colmap",

            "exhaustive_matcher",

            "--database_path",
            database_path,

            "--SiftMatching.use_gpu",
            "0"

        ])


        # =====================================================
        # MAPPER
        # =====================================================

        run_command([

            "colmap",

            "mapper",

            "--database_path",
            database_path,

            "--image_path",
            images_dir,

            "--output_path",
            sparse_dir

        ])


        # =====================================================
        # FIND MODEL
        # =====================================================

        model_dir = find_sparse_model(
            sparse_dir
        )


        if model_dir is None:

            raise Exception(

                "COLMAP could not reconstruct "
                "the cameras. Try taking the "
                "three photos with more overlap."

            )


        print(
            "Sparse reconstruction:",
            model_dir
        )


        # =====================================================
        # UNDISTORT
        # =====================================================

        run_command([

            "colmap",

            "image_undistorter",

            "--image_path",
            images_dir,

            "--input_path",
            model_dir,

            "--output_path",
            dense_dir,

            "--output_type",
            "COLMAP"

        ])


        # =====================================================
        # PATCH MATCH
        # =====================================================

        run_command([

            "colmap",

            "patch_match_stereo",

            "--workspace_path",
            dense_dir,

            "--workspace_format",
            "COLMAP",

            "--PatchMatchStereo.geom_consistency",
            "true",

            "--PatchMatchStereo.gpu_index",
            "-1"

        ])


        # =====================================================
        # FUSION
        # =====================================================

        fused_path = os.path.join(
            dense_dir,
            "fused.ply"
        )


        run_command([

            "colmap",

            "stereo_fusion",

            "--workspace_path",
            dense_dir,

            "--workspace_format",
            "COLMAP",

            "--output_path",
            fused_path

        ])


        if not os.path.exists(
            fused_path
        ):

            raise Exception(
                "COLMAP did not generate fused.ply"
            )


        print(
            "Dense point cloud generated."
        )


        # =====================================================
        # OPTIONAL DEPTH INFORMATION
        # =====================================================

        if depth_paths:

            print(
                "Depth maps available for "
                "additional processing."
            )


        # =====================================================
        # POINT CLOUD → MESH
        # =====================================================

        mesh_path = os.path.join(
            temp_dir,
            "model.ply"
        )


        create_mesh(
            fused_path,
            mesh_path
        )


        # =====================================================
        # MESH → GLB
        # =====================================================

        glb_path = os.path.join(
            temp_dir,
            "objectscan.glb"
        )


        convert_to_glb(
            mesh_path,
            glb_path
        )


        if not os.path.exists(
            glb_path
        ):

            raise Exception(
                "GLB was not generated."
            )


        print("")
        print(
            "=========================================="
        )

        print(
            "GLB GENERATED:",
            glb_path
        )

        print(
            "SIZE:",
            os.path.getsize(
                glb_path
            ),
            "bytes"
        )

        print(
            "=========================================="
        )


        # =====================================================
        # RETURN GLB
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
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )

        print(
            "3D PROCESSING ERROR"
        )

        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )


        traceback.print_exc()


        return jsonify({

            "ok":
                False,

            "error":
                str(error)

        }), 500


    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


# =========================================================
# CHECK COLMAP
# =========================================================

def check_colmap():

    try:

        result = subprocess.run(

            [
                "colmap",
                "-h"
            ],

            stdout=subprocess.PIPE,

            stderr=subprocess.STDOUT,

            text=True,

            timeout=10

        )


    except FileNotFoundError as error:

        raise RuntimeError(

            "COLMAP is not installed. "

            "The Render server needs COLMAP "
            "installed at system level."

        ) from error


    except Exception as error:

        raise RuntimeError(

            "Could not execute COLMAP: "
            + str(error)

        ) from error


    if result.returncode != 0:

        raise RuntimeError(
            "COLMAP exists but could not run."
        )


    print(
        "COLMAP available."
    )


# =========================================================
# RUN COMMAND
# =========================================================

def run_command(command):

    print("")
    print(
        "=========================================="
    )

    print(
        "RUNNING:"
    )

    print(
        " ".join(command)
    )

    print(
        "=========================================="
    )


    try:

        result = subprocess.run(

            command,

            stdout=subprocess.PIPE,

            stderr=subprocess.STDOUT,

            text=True,

            timeout=900

        )


    except FileNotFoundError as error:

        raise RuntimeError(

            f"Executable not found: {command[0]}"

        ) from error


    print(
        result.stdout
    )


    if result.returncode != 0:

        raise RuntimeError(

            "Command failed:\n\n"

            + " ".join(command)

            + "\n\n"

            + result.stdout

        )


    return result.stdout


# =========================================================
# FIND SPARSE MODEL
# =========================================================

def find_sparse_model(
    sparse_dir
):

    if not os.path.exists(
        sparse_dir
    ):

        return None


    candidates = []


    for name in os.listdir(
        sparse_dir
    ):

        path = os.path.join(
            sparse_dir,
            name
        )


        if not os.path.isdir(
            path
        ):

            continue


        cameras = os.path.join(
            path,
            "cameras.bin"
        )


        images = os.path.join(
            path,
            "images.bin"
        )


        points = os.path.join(
            path,
            "points3D.bin"
        )


        if (

            os.path.exists(cameras)

            and

            os.path.exists(images)

            and

            os.path.exists(points)

        ):

            candidates.append(
                path
            )


    if not candidates:

        return None


    return candidates[0]


# =========================================================
# POINT CLOUD → MESH
# =========================================================

def create_mesh(
    point_cloud_path,
    mesh_path
):

    print(
        "Loading point cloud..."
    )


    pcd = o3d.io.read_point_cloud(
        point_cloud_path
    )


    if len(pcd.points) == 0:

        raise Exception(
            "Point cloud is empty."
        )


    print(
        "Points:",
        len(pcd.points)
    )


    # =====================================================
    # DOWNSAMPLE
    # =====================================================

    print(
        "Downsampling point cloud..."
    )


    voxel_size = 0.003


    pcd = pcd.voxel_down_sample(
        voxel_size
    )


    print(
        "Points after downsample:",
        len(pcd.points)
    )


    if len(pcd.points) < 100:

        raise Exception(
            "Not enough points to create mesh."
        )


    # =====================================================
    # NORMALS
    # =====================================================

    print(
        "Estimating normals..."
    )


    pcd.estimate_normals(

        search_param=
        o3d.geometry.KDTreeSearchParamHybrid(

            radius=0.05,

            max_nn=30

        )

    )


    try:

        pcd.orient_normals_consistent_tangent_plane(
            20
        )

    except Exception:

        print(
            "Could not consistently orient normals."
        )


    # =====================================================
    # POISSON
    # =====================================================

    print(
        "Creating mesh..."
    )


    mesh, densities = (

        o3d.geometry.TriangleMesh
        .create_from_point_cloud_poisson(

            pcd,

            depth=8,

            width=0,

            scale=1.1,

            linear_fit=True

        )

    )


    # =====================================================
    # REMOVE LOW DENSITY
    # =====================================================

    densities = np.asarray(
        densities
    )


    if len(densities) > 0:

        threshold = np.quantile(
            densities,
            0.03
        )


        vertices_to_remove = (

            densities <
            threshold

        )


        mesh.remove_vertices_by_mask(
            vertices_to_remove
        )


    # =====================================================
    # CLEAN
    # =====================================================

    mesh.remove_degenerate_triangles()

    mesh.remove_duplicated_triangles()

    mesh.remove_duplicated_vertices()

    mesh.remove_non_manifold_edges()


    # =====================================================
    # SAVE
    # =====================================================

    success = o3d.io.write_triangle_mesh(

        mesh_path,

        mesh,

        write_vertex_colors=True

    )


    if not success:

        raise Exception(
            "Could not save mesh."
        )


    print(
        "Mesh saved:",
        mesh_path
    )


# =========================================================
# MESH → GLB
# =========================================================

def convert_to_glb(
    mesh_path,
    glb_path
):

    print(
        "Converting mesh to GLB..."
    )


    mesh = trimesh.load(
        mesh_path
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


    print(
        "GLB saved:",
        glb_path
    )


# =========================================================
# SERVER
# =========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=PORT

    )
