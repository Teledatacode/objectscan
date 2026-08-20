from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

import os
import tempfile

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():

    return "ObjectScan 3D server is running."


@app.route("/process-3d", methods=["POST"])
def process_3d():

    print("================================")
    print("3D PROCESSING REQUEST")
    print("================================")

    # -----------------------------------------
    # COMPROBAR LAS 3 FOTOS
    # -----------------------------------------

    image1 = request.files.get("image1")
    image2 = request.files.get("image2")
    image3 = request.files.get("image3")

    if not image1:
        return jsonify({
            "ok": False,
            "error": "Missing image1"
        }), 400

    if not image2:
        return jsonify({
            "ok": False,
            "error": "Missing image2"
        }), 400

    if not image3:
        return jsonify({
            "ok": False,
            "error": "Missing image3"
        }), 400


    print("image1:", image1.filename)
    print("image2:", image2.filename)
    print("image3:", image3.filename)


    # -----------------------------------------
    # DIRECTORIO TEMPORAL
    # -----------------------------------------

    temp_dir = tempfile.mkdtemp()

    image1_path = os.path.join(
        temp_dir,
        "image1.jpg"
    )

    image2_path = os.path.join(
        temp_dir,
        "image2.jpg"
    )

    image3_path = os.path.join(
        temp_dir,
        "image3.jpg"
    )


    image1.save(image1_path)
    image2.save(image2_path)
    image3.save(image3_path)


    print("Photos saved:")
    print(image1_path)
    print(image2_path)
    print(image3_path)


    # -----------------------------------------
    # AQUÍ VA EL PROCESAMIENTO 3D
    # -----------------------------------------

    try:

        glb_path = create_3d_model(
            image1_path,
            image2_path,
            image3_path
        )

    except Exception as error:

        print(
            "3D PROCESSING ERROR:",
            error
        )

        return jsonify({
            "ok": False,
            "error": str(error)
        }), 500


    # -----------------------------------------
    # DEVOLVER GLB
    # -----------------------------------------

    if not os.path.exists(glb_path):

        return jsonify({
            "ok": False,
            "error": "GLB was not generated."
        }), 500


    print(
        "GLB generated:",
        glb_path
    )


    return send_file(
        glb_path,
        mimetype="model/gltf-binary",
        as_attachment=False,
        download_name="objectscan.glb"
    )


def create_3d_model(
    image1,
    image2,
    image3
):

    """
    AQUÍ construiremos el modelo 3D
    utilizando las tres fotografías.
    """

    print("Creating 3D model...")

    # TEMPORAL:
    # todavía no hacemos reconstrucción.

    raise NotImplementedError(
        "3D reconstruction engine not implemented yet."
    )


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
