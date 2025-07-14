
#!/usr/bin/env python3

"""
predict_vibration.py
--------------------
Loads mobilenet_vibration_classifier.h5 and classifies .npy vibration signals.
Usage:
    python predict_vibration.py  my_signal.npy
    python predict_vibration.py  folder_with_many_npy/
"""
import os
import sys
from io import BytesIO

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------- constants ----------
IMG_H, IMG_W = 128, 128
FS           = 2_500
MODEL_PATH   = "mobilenet_vibration_classifier.keras"

# --- custom Lambda used when the model was saved ---
def gray2rgb(x):
    return tf.image.grayscale_to_rgb(x)

# ---------- load model ----------
model = load_model(MODEL_PATH, compile=False,
                   custom_objects={'gray2rgb': gray2rgb})

# ---------- helper: 1‑D signal → spectrogram tensor ----------
def signal_to_tensor(signal: np.ndarray) -> np.ndarray:
    S = librosa.stft(signal, n_fft=256, hop_length=128)
    S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    fig = plt.figure(figsize=(2.5, 2.5), dpi=IMG_H / 2.5)
    librosa.display.specshow(S_dB, sr=FS, hop_length=128, cmap="gray")
    plt.axis("off"); plt.tight_layout(pad=0)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig); buf.seek(0)

    img = Image.open(buf).convert("L").resize((IMG_W, IMG_H))
    arr = (np.array(img, dtype=np.float32) / 255.0)[..., None]   # (128,128,1)
    return arr[None, ...]                                        # (1,128,128,1)

# ---------- prediction ----------
def predict_file(npy_path: str) -> None:
    signal = np.load(npy_path)
    prob = float(model.predict(signal_to_tensor(signal), verbose=0)[0, 0])
    label = "NORMAL ✅" if prob < 0.5 else "FAULT ⚠️"
    print(f"{os.path.basename(npy_path):<25} → {label}  (confidence: {prob:.3f})")

# ---------- CLI ----------
def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: python predict_vibration.py <.npy file | folder>")

    target = sys.argv[1]
    if os.path.isfile(target) and target.lower().endswith(".npy"):
        predict_file(target)
    elif os.path.isdir(target):
        files = sorted(f for f in (os.path.join(target, n) for n in os.listdir(target))
                       if f.lower().endswith(".npy"))
        if not files:
            sys.exit("❌ No .npy files found in the folder.")
        for f in files:
            predict_file(f)
    else:
        sys.exit("❌ Provide a valid .npy file or directory of .npy files.")

if __name__ == "__main__":
    main()










# import os
# import sys
# import numpy as np
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# from PIL import Image
# import tensorflow as tf

# IMG_HEIGHT, IMG_WIDTH = 128, 128
# FS = 2500
# MODEL_PATH = "mobilenet_vibration_classifier-1.h5"

# def gray_to_rgb(x):
#     return tf.image.grayscale_to_rgb(x)

# # Load model with custom_objects to handle gray_to_rgb Lambda layer
# model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'gray_to_rgb': gray_to_rgb}, compile=False)

# def npy_to_spectrogram_image(signal):
#     S = librosa.stft(signal, n_fft=256, hop_length=128)
#     S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)
#     fig = plt.figure(figsize=(2.5, 2.5))
#     librosa.display.specshow(S_dB, sr=FS, hop_length=128, cmap='gray')
#     plt.axis('off')
#     plt.tight_layout()
#     from io import BytesIO
#     buf = BytesIO()
#     plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
#     plt.close(fig)
#     buf.seek(0)
#     img = Image.open(buf).convert('L').resize((IMG_WIDTH, IMG_HEIGHT))
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=(0, -1))
#     return img_array

# def predict_npy_file(npy_path):
#     signal = np.load(npy_path)
#     image_tensor = npy_to_spectrogram_image(signal)
#     prob = model.predict(image_tensor)[0][0]
#     label = "NORMAL ✅" if prob < 0.5 else "FAULT ⚠️"
#     return label, round(float(prob), 3)

# def main(path):
#     if not path:
#         print("Usage: python predictor.py / predict_vibration.py <.npy file | folder>")
#         return

#     if os.path.isfile(path) and path.endswith('.npy'):
#         label, conf = predict_npy_file(path)
#         print(f"{os.path.basename(path)} → {label} (confidence: {conf})")
#     elif os.path.isdir(path):
#         npy_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.npy')]
#         if not npy_files:
#             print("❌ No .npy files found in the folder.")
#             return
#         for f in sorted(npy_files):
#             label, conf = predict_npy_file(f)
#             print(f"{os.path.basename(f)} → {label} (confidence: {conf})")
#     else:
#         print("Usage: python predict_vibration.py <.npy file | folder>")

# if __name__ == "__main__":
#     main(sys.argv[1] if len(sys.argv) > 1 else "")
