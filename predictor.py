import os
import sys
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable  # Required for Lambda

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings

# Constants
FS = 2500  # Sampling frequency
IMG_HEIGHT, IMG_WIDTH = 128, 128  # Image dimensions for model input
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mobilenet_vibration_classifier1.keras")

# ========== REGISTERED FUNCTION ========== #
@register_keras_serializable()
def gray_to_rgb(x):
    """Convert grayscale image to RGB by duplicating channels"""
    return tf.image.grayscale_to_rgb(x)

# ========== SIGNAL PROCESSING ========== #
def npy_to_mel_spectrogram_image(signal):
    """
    Convert 1D vibration signal to mel spectrogram image
    Args:
        signal: 1D numpy array containing vibration data
    Returns:
        Preprocessed image tensor ready for model prediction
    """
    # Compute mel spectrogram (note 'y=' keyword argument)
    S = librosa.feature.melspectrogram(y=signal, sr=FS, n_mels=128, fmax=FS//2)
    
    # Convert to dB scale
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Normalize to 0-1 range
    img = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
    
    # Flip vertically (librosa displays low freqs at bottom)
    img = np.flip(img, axis=0)
    
    # Add channel dimension (grayscale)
    img = np.expand_dims(img, axis=-1)
    
    # Resize to model input dimensions
    img_resized = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH]).numpy()
    
    # Add batch dimension and ensure float32
    return np.expand_dims(img_resized, axis=0).astype(np.float32)

# ========== MODEL HANDLING ========== #
def load_model():
    """
    Load the pre-trained vibration classifier model
    Returns:
        Loaded Keras model
    Raises:
        SystemExit: If model file is not found
    """
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found at: {MODEL_PATH}")
        print("Please ensure the model file exists at this location.")
        sys.exit(1)
        
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"gray_to_rgb": gray_to_rgb},
            compile=False
        )
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {str(e)}")
        sys.exit(1)

# ========== PREDICTION ========== #
def predict_npy_file(model, npy_path):
    """
    Make prediction on a single .npy file
    Args:
        model: Loaded Keras model
        npy_path: Path to .npy file
    Returns:
        tuple: (prediction_label, confidence_score)
    """
    try:
        signal = np.load(npy_path)
        if len(signal.shape) != 1:
            print(f"[WARNING] Expected 1D signal in {os.path.basename(npy_path)}, got shape {signal.shape}")
            
        image_tensor = npy_to_mel_spectrogram_image(signal)
        prob = model.predict(image_tensor, verbose=0)[0][0]
        
        label = "NORMAL" if prob < 0.5 else "FAULT"
        confidence = float(prob) if label == "FAULT" else 1 - float(prob)
        
        return label, round(confidence, 4)
        
    except Exception as e:
        print(f"[ERROR] Failed to process {os.path.basename(npy_path)}: {str(e)}")
        return "ERROR", 0.0

# ========== MAIN FUNCTION ========== #
def main(input_path):
    """
    Main entry point for vibration classification
    Args:
        input_path: Path to .npy file or directory containing .npy files
    """
    if not input_path:
        print("Usage: python predict_vibration_v2.py <.npy file or folder>")
        return
        
    input_path = os.path.abspath(input_path)
    
    try:
        model = load_model()
    except Exception as e:
        print(f"[ERROR] Model loading failed: {str(e)}")
        return

    # Single file case
    if os.path.isfile(input_path) and input_path.lower().endswith('.npy'):
        label, confidence = predict_npy_file(model, input_path)
        print(f"{os.path.basename(input_path)} → {label} (confidence: {confidence:.3f})")
        return f"{label}"
    
    # Directory case
    elif os.path.isdir(input_path):
        npy_files = sorted(
            f for f in os.listdir(input_path) 
            if f.lower().endswith('.npy')
        )
        
        if not npy_files:
            print(f"[WARNING] No .npy files found in: {input_path}")
            return
            
        print(f"Found {len(npy_files)} .npy files in directory")
        print("=" * 60)
        
        for f in npy_files:
            full_path = os.path.join(input_path, f)
            label, confidence = predict_npy_file(model, full_path)
            print(f"{f} → {label} (confidence: {confidence:.3f})")
            return f"{label}"
    
    else:
        print(f"[ERROR] Invalid path: {input_path}")
        print("Please provide either a .npy file or directory containing .npy files")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Vibration Classifier Tool")
        print("Usage: python predict_vibration_v2.py <path_to_npy_file_or_directory>")
        sys.exit(0)
        
    main(sys.argv[1])