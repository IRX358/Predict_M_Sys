# predictor.py

import sys
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import io
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import json
import os

# Configuration - adjust as needed
MODEL_PATH = 'mobilenet_vibration_classifier1.keras'
IMG_HEIGHT = 128
IMG_WIDTH = 128
METRICS_JSON = 'model_metrics.json'

def signal_to_spectrogram_image(sig, fs=2500):
    # Convert 1D signal to grayscale spectrogram image (matching training preprocessing)
    S = librosa.stft(sig, n_fft=256, hop_length=128)
    S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    plt.figure(figsize=(2.5,2.5))
    librosa.display.specshow(S_dB, sr=fs, hop_length=128, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    img = Image.open(buf).convert('L')  # grayscale
    buf.close()
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img)
    return img_array

def predict_single(filepath, model):
    sig = np.load(filepath)
    img = signal_to_spectrogram_image(sig)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)  # add grayscale channel
    img = np.expand_dims(img, axis=0)   # add batch dim
    pred_prob = model.predict(img)[0][0]
    return 'normal' if pred_prob <= 0.5 else 'faulty'

def load_metrics_json():
    if os.path.exists(METRICS_JSON):
        with open(METRICS_JSON, 'r') as f:
            return json.load(f)
    else:
        return None

def compute_and_save_metrics(model, test_folder):
    """
    Compute metrics on a directory structured as:
    test_folder/
       normal/*.npy
       fault/*.npy

    Saves metrics json to METRICS_JSON file.
    """

    classes = ['normal', 'fault']
    label_map = {cls: i for i, cls in enumerate(classes)}
    X_images = []
    y_true = []

    for cls in classes:
        cls_folder = os.path.join(test_folder, cls)
        if not os.path.isdir(cls_folder):
            continue
        for fname in os.listdir(cls_folder):
            if fname.endswith('.npy'):
                filepath = os.path.join(cls_folder, fname)
                sig = np.load(filepath)
                img = signal_to_spectrogram_image(sig)
                X_images.append(img)
                y_true.append(label_map[cls])

    if len(X_images) == 0:
        raise RuntimeError("No test files found.")

    X_images = np.array(X_images).astype(np.float32) / 255.0
    X_images = np.expand_dims(X_images, axis=-1)  # grayscale channel

    y_true = np.array(y_true)

    y_pred_probs = model.predict(X_images).flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()  # convert to nested list for JSON

    metrics = {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'confusion_matrix': cm
    }

    # Save to JSON file
    with open(METRICS_JSON, 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics

def main():
    # Usage:
    #   For single prediction:
    #     python predictor.py path/to/single_file.npy
    #   To compute metrics on a test folder:
    #     python predictor.py compute_metrics path/to/test_folder

    model = tf.keras.models.load_model(MODEL_PATH)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  For single prediction:")
        print("    python predictor.py path/to/file.npy")
        print("  To compute metrics on test data folder:")
        print("    python predictor.py compute_metrics path/to/test_folder")
        sys.exit(1)

    if sys.argv[1] == 'compute_metrics':
        if len(sys.argv) != 3:
            print("Error: Test folder path required.")
            sys.exit(1)
        test_folder = sys.argv[2]
        metrics = compute_and_save_metrics(model, test_folder)
        print("Metrics computed and saved to", METRICS_JSON)
        print(json.dumps(metrics, indent=4))
    else:
        # Single prediction case
        file_path = sys.argv[1]
        pred = predict_single(file_path, model)
        print(pred)  # For your flask app to capture this output as normal/faulty

if __name__ == '__main__':
    main()