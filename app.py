from flask import Flask, request, jsonify
import cv2
import numpy as np
import joblib
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model
MODEL_FILE = "paddy_phase_classifier_rf.joblib"
model = joblib.load(MODEL_FILE)

## Function to extract color histogram features
def extract_color_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

@app.route('/predict', methods=['POST'])
def predict_paddy_phase():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    image_path = os.path.join("uploads", file.filename)
    file.save(image_path)
    image = cv2.imread(image_path)
    if image is None:
        return jsonify({'error': 'Invalid image format'}), 400
    image = cv2.resize(image, (128, 128))
    features = extract_color_histogram(image).reshape(1, -1)
    prediction = model.predict(features)[0]
    os.remove(image_path)
    return jsonify({'category': prediction})

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
