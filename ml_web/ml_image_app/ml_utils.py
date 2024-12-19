# ml_utils.py
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your model
model = tf.saved_model.load(r'D:\Pribadi\Sidehustle\p24\ml_image_annotation\ssd_mobilenet_v2_coco_2018_03_29\saved_model')

# Object detection function
def detect_objects(image_path):
    img = np.array(Image.open(image_path))
    input_tensor = tf.convert_to_tensor(img)
    detections = model(input_tensor)
    return detections

# Train model function (simplified for demo purposes)
def train_model():
    # Implement your training logic (e.g., fine-tuning or retraining)
    print("Training model...")
    # model.fit(training_data)  # Add your actual training loop here
    return "Training completed"
