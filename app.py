from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = "depression_model.h5"
model = load_model(MODEL_PATH)

# Image settings (same as training)
IMG_HEIGHT, IMG_WIDTH = 128, 128

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"

        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode="grayscale")
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)[0][0]
        result = "Depressed" if prediction > 0.5 else "Not Depressed"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
