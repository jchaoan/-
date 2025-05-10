from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
from PIL import Image
import io
from flask import render_template
app = Flask(__name__)

# 載入模型
model = YOLO("models/初試v1.pt")

@app.route('/')
def index():
    return "YOLOv8 Flask API is running."
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image_file = request.files['image']
    image = Image.open(image_file.stream)

    results = model(image)
    predictions = results[0].boxes.data.tolist()  # x1, y1, x2, y2, confidence, class_id

    response = []
    for box in predictions:
        x1, y1, x2, y2, conf, class_id = box
        response.append({
            "class_id": int(class_id),
            "confidence": float(conf),
            "bbox": [float(x1), float(y1), float(x2), float(y2)]
        })

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
