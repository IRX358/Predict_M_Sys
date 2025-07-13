import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import argparse
import io
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# === Configuration === generator and sample files\4. Bearings\IMS\1st_test
fs = 2500
img_size = (128, 128)

from keras.utils import register_keras_serializable

@register_keras_serializable()
def gray_to_rgb(x):
    import tensorflow as tf
    return tf.image.grayscale_to_rgb(x)

model_path = "mobilenet_vibration_classifier.h5"
model = load_model(model_path,compile=False)

def generate_spectrogram_image(signal):
    S = librosa.stft(signal, n_fft=256, hop_length=128)
    S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    fig = plt.figure(figsize=(2.5, 2.5))
    librosa.display.specshow(S_dB, sr=fs, hop_length=128, cmap='gray')
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert('L').resize(img_size)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # (1, 128, 128, 1)
    return img_array

def predict_file(filepath):
    try:
        signal = np.load(filepath)
        spectrogram = generate_spectrogram_image(signal)
        pred = model.predict(spectrogram)[0][0]
        label = "NORMAL ✅" if pred >= 0.5 else "FAULT ⚠"
        print(f"{filepath} → {label} (Confidence: {round(float(pred), 2)})")
    except Exception as e:
        print(f"❌ Failed to process {filepath}: {e}")

def process_path(path):
    if path.endswith(".npy"):
        predict_file(path)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".npy"):
                    predict_file(os.path.join(root, file))
    else:
        print("❗ Invalid input path: Provide a .npy file or folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict bearing health from .npy vibration files.")
    parser.add_argument("input_path", help="Path to a .npy file or a folder containing .npy files")
    args = parser.parse_args()
    process_path(args.input_path)
