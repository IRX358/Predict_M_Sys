import numpy as np
import json
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# === Register custom function used in Lambda layer ===
def gray_to_rgb(x):
    return tf.image.grayscale_to_rgb(x)

# === Load model with custom_objects ===
model = load_model("mobilenet_vibration_classifier.h5", custom_objects={'gray_to_rgb': gray_to_rgb})

# === Load test data ===
X_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

# === Predict ===
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype("int32").flatten()

# === Classification report ===
report = classification_report(y_test, y_pred_classes, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# === Prepare metrics JSON ===
metrics = {
    "accuracy": round(report["accuracy"], 4),
    "precision": round(report["1"]["precision"], 4),
    "recall": round(report["1"]["recall"], 4),
    "f1_score": round(report["1"]["f1-score"], 4),
    "confusion_matrix": conf_matrix.tolist()
}

# === Save to file ===
with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ metrics.json created successfully.")