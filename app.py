from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import traceback


# =========================================================
# APP
# =========================================================

app = Flask(__name__)


# =========================================================
# CORS
# =========================================================

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*"
        }
    }
)


# =========================================================
# HOME
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
# TEST
# =========================================================

@app.route("/test", methods=["GET"])
def test():

    return jsonify({

        "ok": True,

        "message":
            "POST endpoint available"

    })


# =========================================================
# PROCESS 3D
#
# TEMPORALMENTE SOLO COMPRUEBA
# QUE LAS 3 IMÁGENES LLEGUEN.
# =========================================================

@app.route(
    "/process-3d",
    methods=["POST"]
)
def process_3d():

    try:

        print("")
        print("================================")
        print("OBJECTSCAN POST TEST")
        print("================================")


        # -------------------------------------------------
        # RECIBIR ARCHIVOS
        # -------------------------------------------------

        image1 = request.files.get(
            "image1"
        )

        image2 = request.files.get(
            "image2"
        )

        image3 = request.files.get(
            "image3"
        )


        print(
            "image1:",
            image1
        )

        print(
            "image2:",
            image2
        )

        print(
            "image3:",
            image3
        )


        # -------------------------------------------------
        # VALIDAR
        # -------------------------------------------------

        if not image1:

            return jsonify({

                "ok": False,

                "error":
                    "Missing image1"

            }), 400


        if not image2:

            return jsonify({

                "ok": False,

                "error":
                    "Missing image2"

            }), 400


        if not image3:

            return jsonify({

                "ok": False,

                "error":
                    "Missing image3"

            }), 400


        # -------------------------------------------------
        # INFORMACIÓN
        # -------------------------------------------------

        print(
            "Files received:"
        )

        print(
            "  image1:",
            image1.filename
        )

        print(
            "  image2:",
            image2.filename
        )

        print(
            "  image3:",
            image3.filename
        )


        # -------------------------------------------------
        # GUARDAR TEMPORALMENTE
        # -------------------------------------------------

        upload_dir = os.path.join(
            "/tmp",
            "objectscan_test"
        )


        os.makedirs(
            upload_dir,
            exist_ok=True
        )


        path1 = os.path.join(
            upload_dir,
            "image1.jpg"
        )

        path2 = os.path.join(
            upload_dir,
            "image2.jpg"
        )

        path3 = os.path.join(
            upload_dir,
            "image3.jpg"
        )


        image1.save(
            path1
        )

        image2.save(
            path2
        )

        image3.save(
            path3
        )


        # -------------------------------------------------
        # TAMAÑOS
        # -------------------------------------------------

        size1 = os.path.getsize(
            path1
        )

        size2 = os.path.getsize(
            path2
        )

        size3 = os.path.getsize(
            path3
        )


        print(
            "Image sizes:"
        )

        print(
            "  image1:",
            size1,
            "bytes"
        )

        print(
            "  image2:",
            size2,
            "bytes"
        )

        print(
            "  image3:",
            size3,
            "bytes"
        )


        # -------------------------------------------------
        # RESPUESTA
        # -------------------------------------------------

        print(
            "Images received correctly."
        )


        return jsonify({

            "ok": True,

            "message":
                "Images received correctly",

            "images": {

                "image1": {
                    "filename":
                        image1.filename,

                    "size":
                        size1
                },

                "image2": {
                    "filename":
                        image2.filename,

                    "size":
                        size2
                },

                "image3": {
                    "filename":
                        image3.filename,

                    "size":
                        size3
                }

            }

        })


    except Exception as error:

        print("")
        print("================================")
        print("POST TEST ERROR")
        print("================================")


        traceback.print_exc()


        return jsonify({

            "ok": False,

            "error":
                str(error)

        }), 500


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
