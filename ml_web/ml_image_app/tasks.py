from .ml_utils import train_model, detect_objects
from .models import ImageUpload

# @shared_task
def process_image(image_id):
    image = ImageUpload.objects.get(id=image_id)
    detections = detect_objects(image.image.path)
    return detections

# @shared_task
def retrain_model():
    # This would trigger model training asynchronously
    train_model()
    return "Training completed"
