from flask import Flask, request, render_template
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

model = load_model("models/pneumonia_model.h5")

IMG_SIZE = 224

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        label = "PNEUMONIA"
        confidence = float(pred)
    else:
        label = "NORMAL"
        confidence = float(1 - pred)

    return label, round(confidence, 2)

@app.route("/", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    filename = None

    if request.method == "POST":
        file = request.files["file"]
        filepath = os.path.join("app/static", file.filename)
        file.save(filepath)

        label, confidence = predict_image(filepath)

        result = label
        filename = file.filename

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        filename=filename
    )

if __name__ == "__main__":
    app.run()