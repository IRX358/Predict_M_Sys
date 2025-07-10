import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

# === CONFIG ===
img_height, img_width = 128, 128
model_path = "mobilenet_vibration_classifier.h5"  # Put the model in your project root

# === Load model once ===
model = load_model(model_path)

def preprocess_image(img_path):
    img = Image.open(img_path).convert('L').resize((img_width, img_height))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dim
    return np.expand_dims(img_array, axis=0)  # Add batch dim

def predict(filepath):
    img_tensor = preprocess_image(filepath)
    pred = model.predict(img_tensor)[0][0]
    return "FAULT ⚠️" if pred > 0.5 else "NORMAL ✅"

# If run directly from terminal
if __name__ == "_main_":
    import sys
    filepath = sys.argv[1]
    print(predict(filepath))




# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     try:
#         processed_input = process_signal_to_image(file)
#         pred_prob = model.predict(processed_input)[0][0]
#         label = "NORMAL ✅" if pred_prob >= 0.5 else "FAULT ⚠️"

#         return jsonify({'prediction': label, 'confidence': round(float(pred_prob), 2)})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)
