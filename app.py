import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # hide TF INFO/WARNING/ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    # disable oneDNN logs

from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load TFLite model
TFLITE_MODEL_PATH = "depression_model.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Image settings (same as training)
IMG_HEIGHT, IMG_WIDTH = 128, 128

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        # Validate file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return "Invalid file type. Please upload PNG/JPG/JPEG.", 400

        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        try:
            # Preprocess image
            img = image.load_img(filepath, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode="grayscale")
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Run inference
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

            # Convert prediction to label
            result = "Depressed" if prediction > 0.5 else "Not Depressed"

        except Exception as e:
            return f"Error processing image: {str(e)}", 500

    return render_template("index.html", result=result)

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
