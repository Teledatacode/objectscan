from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # permite todos los orígenes

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
VIEWER_FOLDER = "viewer"
os.makedirs(VIEWER_FOLDER, exist_ok=True)

# Endpoint para subir fotos
@app.route("/upload", methods=["POST"])
def upload_photos():
    photos = request.files
    saved_files = []
    filenames = []

    for key in photos:
        file = photos[key]
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        saved_files.append(path)
        filenames.append(filename)

    # Generar un HTML de visor 360 simple
    viewer_html_path = os.path.join(VIEWER_FOLDER, "viewer360.html")
    with open(viewer_html_path, "w") as f:
        f.write("<!DOCTYPE html>\n<html lang='es'>\n<head>\n<meta charset='UTF-8'>\n")
        f.write("<meta name='viewport' content='width=device-width,initial-scale=1.0'>\n")
        f.write("<title>Visor 360</title>\n<style>body{margin:0;background:black;display:flex;justify-content:center;align-items:center;height:100vh;}img{max-width:80vw;max-height:80vh;}</style>\n")
        f.write("</head>\n<body>\n")
        f.write("<img id='viewer' src='/viewer_image/{}'>\n".format(filenames[0]))
        f.write("<script>\n")
        f.write("const images = {};\n".format(filenames))
        f.write("let index=0;\n")
        f.write("const viewer = document.getElementById('viewer');\n")
        f.write("document.body.addEventListener('click', ()=>{index=(index+1)%images.length;viewer.src='/viewer_image/'+images[index];});\n")
        f.write("</script>\n</body>\n</html>")
    
    # Devolver URL accesible para frontend
    viewer_url = "/viewer/viewer360.html"
    return jsonify({"viewer_url": viewer_url})

# Endpoint para servir imágenes en el visor
@app.route("/viewer_image/<filename>")
def serve_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Endpoint para servir el HTML del visor
@app.route("/viewer/<path:path>")
def serve_viewer(path):
    return send_from_directory(VIEWER_FOLDER, path)

if __name__=="__main__":
    app.run(debug=True)
