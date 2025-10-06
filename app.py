from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # permite todos los orígenes, para pruebas



UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_photos():
    photos = request.files
    saved_files = []
    for key in photos:
        file = photos[key]
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        saved_files.append(path)
    
    # Aquí iría la lógica para generar un visor 360 con las fotos
    # Por ejemplo, un HTML con slider o librería 360
    viewer_url = "https://tuapp.onrender.com/viewer360.html"
    return jsonify({"viewer_url": viewer_url})

if __name__=="__main__":
    app.run(debug=True)
