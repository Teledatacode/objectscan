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
# HEALTH CHECK
# =========================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "ok": True,
        "service": "ObjectScan 3D",
        "status": "running"
    })


# =========================================================
# 3D PROCESSING
# =========================================================

@app.route("/process-3d", methods=["POST"])
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
        # 1. RECIBIR IMÁGENES
        # =====================================================

        image1 = request.files.get("image1")
        image2 = request.files.get("image2")
        image3 = request.files.get("image3")


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
        # 2. COMPROBAR ARCHIVOS
        # =====================================================

        for path in image_paths:

            if not os.path.exists(path):

                raise Exception(
                    f"Image was not saved: {path}"
                )


            print(
                path,
                os.path.getsize(path),
                "bytes"
            )


        # =====================================================
        # 3. PREPARAR COLMAP
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
        # 4. FEATURE EXTRACTION
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
        # 5. FEATURE MATCHING
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
        # 6. STRUCTURE FROM MOTION
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
        # 7. ENCONTRAR MODELO
        # =====================================================

        model_dir = find_sparse_model(
            sparse_dir
        )


        if model_dir is None:

            raise Exception(
                "COLMAP could not reconstruct the cameras. "
                "The 3 photos may not contain enough overlapping features."
            )


        print(
            "Sparse reconstruction:",
            model_dir
        )


        # =====================================================
        # 8. IMAGE UNDISTORT
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
        # 9. PATCH MATCH STEREO
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
        # 10. FUSIONAR POINT CLOUD
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
        # 11. POINT CLOUD → MESH
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
        # 12. MESH → GLB
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


        print(
            "=========================================="
        )

        print(
            "GLB GENERATED:",
            glb_path
        )

        print(
            "SIZE:",
            os.path.getsize(glb_path),
            "bytes"
        )

        print(
            "=========================================="
        )


        # =====================================================
        # 13. DEVOLVER GLB
        # =====================================================

        return send_file(

            glb_path,

            mimetype="model/gltf-binary",

            as_attachment=False,

            download_name="objectscan.glb"

        )


    except Exception as error:

        print("")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("3D PROCESSING ERROR")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        traceback.print_exc()


        return jsonify({

            "ok": False,

            "error":
                str(error)

        }), 500


    finally:

        # =====================================================
        # LIMPIAR TEMPORALES
        # =====================================================

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


# =========================================================
# RUN COMMAND
# =========================================================

def run_command(command):

    print("")
    print(
        "RUNNING:",
        " ".join(command)
    )


    result = subprocess.run(

        command,

        stdout=subprocess.PIPE,

        stderr=subprocess.STDOUT,

        text=True

    )


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


    # -------------------------------------------------------
    # NORMALS
    # -------------------------------------------------------

    pcd.estimate_normals(

        search_param=
        o3d.geometry.KDTreeSearchParamHybrid(

            radius=0.05,

            max_nn=30

        )

    )


    pcd.orient_normals_consistent_tangent_plane(
        20
    )


    # -------------------------------------------------------
    # POISSON RECONSTRUCTION
    # -------------------------------------------------------

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


    # -------------------------------------------------------
    # REMOVE LOW DENSITY AREAS
    # -------------------------------------------------------

    densities = np.asarray(
        densities
    )


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


    # -------------------------------------------------------
    # CLEAN
    # -------------------------------------------------------

    mesh.remove_degenerate_triangles()

    mesh.remove_duplicated_triangles()

    mesh.remove_duplicated_vertices()

    mesh.remove_non_manifold_edges()


    # -------------------------------------------------------
    # SAVE
    # -------------------------------------------------------

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
