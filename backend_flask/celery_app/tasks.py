from .celery import app
from ultralytics import YOLO
import os
import yaml
from datetime import datetime
from bson import ObjectId
from gridfs import GridFS
import logging

@app.task(bind=True)
def train_yolo_model(self, training_id, dataset_config, training_params, base_path="/tmp/yolo_training"):
    """Celery task for training YOLO model"""
    # Use the shared GridFS from the Celery app
    mongo_db = app.mongo_db
    fs = app.fs

    try:
        # Logging setup
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Update status to preparing
        mongo_db.trainings.update_one(
            {'_id': ObjectId(training_id)},
            {'$set': {
                'status': 'preparing',
                'progress': 0,
                'message': 'Preparing dataset'
            }}
        )

        # Prepare dataset
        dataset_path = prepare_dataset(training_id, dataset_config, base_path, fs, logger)
        
        # Initialize model
        model = YOLO('yolov8n.pt')
        
        # Update status to training
        mongo_db.trainings.update_one(
            {'_id': ObjectId(training_id)},
            {'$set': {
                'status': 'training',
                'message': 'Training started'
            }}
        )

        # Training callback to update progress
        def on_train_epoch_end(trainer):
            current_epoch = trainer.epoch
            total_epochs = training_params.get('epochs', 100)
            progress = (current_epoch / total_epochs) * 100
            
            metrics = trainer.metrics
            mongo_db.trainings.update_one(
                {'_id': ObjectId(training_id)},
                {'$set': {
                    'progress': progress,
                    'message': f'Epoch {current_epoch}/{total_epochs}',
                    'metrics': {
                        'mAP': float(metrics.get('metrics/mAP50-95(B)', 0)),
                        'precision': float(metrics.get('metrics/precision(B)', 0)),
                        'recall': float(metrics.get('metrics/recall(B)', 0))
                    }
                }}
            )
            
            # Update Celery task state
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': current_epoch,
                    'total': total_epochs,
                    'progress': progress
                }
            )

        # Start training
        model.train(
            data=os.path.join(dataset_path, 'dataset.yaml'),
            epochs=training_params.get('epochs', 100),
            batch=training_params.get('batch_size', 16),
            imgsz=training_params.get('image_size', 640)
        )

        # Save final model to GridFS
        model_path = os.path.join(dataset_path, 'runs/detect/train/weights/best.pt')
        with open(model_path, 'rb') as f:
            fs.put(
                f.read(),
                filename=f'yolo_model_{training_id}.pt',
                training_id=training_id
            )

        # Update status to completed
        mongo_db.trainings.update_one(
            {'_id': ObjectId(training_id)},
            {'$set': {
                'status': 'completed',
                'progress': 100,
                'message': 'Training completed successfully'
            }}
        )

        return {'status': 'completed'}

    except Exception as e:
        # Update status to failed if error
        mongo_db.trainings.update_one(
            {'_id': ObjectId(training_id)},
            {'$set': {
                'status': 'failed',
                'message': str(e)
            }}
        )
        raise

def prepare_dataset(training_id, dataset_config, base_path, fs, logger):
    """Prepare dataset structure for YOLO training"""
    dataset_path = os.path.join(base_path, training_id)
    os.makedirs(dataset_path, exist_ok=True)
    
    # Create directory structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(dataset_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, split, 'labels'), exist_ok=True)

    # Tracking processed images
    processed_images = 0
    total_images = len(dataset_config['images'])

    # Get images from GridFS and save
    for image_id in dataset_config['images']:
        try:
            # Debug logging
            logger.info(f"Attempting to retrieve image: {image_id}")
            
            # Explicitly check if file exists before retrieving
            file_exists = fs.exists(ObjectId(image_id))
            if not file_exists:
                logger.warning(f"File {image_id} does not exist in GridFS")
                continue

            # Retrieve file
            file = fs.get(ObjectId(image_id))
            image_data = file.read()
            
            # Save image
            image_path = os.path.join(dataset_path, 'train/images', f'{image_id}.jpg')
            with open(image_path, 'wb') as f:
                f.write(image_data)
                
            # Create dummy label for now
            label_path = os.path.join(dataset_path, 'train/labels', f'{image_id}.txt')
            with open(label_path, 'w') as f:
                f.write('0 0.5 0.5 0.3 0.3\n')  # Dummy label
            
            processed_images += 1
            logger.info(f"Processed image {image_id}")

        except Exception as e:
            logger.error(f"Error processing image {image_id}: {e}")

    # Validate processed images
    if processed_images == 0:
        raise ValueError(f"No images could be processed. Total images attempted: {total_images}")

    # Create dataset.yaml
    dataset_yaml = {
        'path': dataset_path,
        'train': 'train/images',
        'val': 'train/images',  # Using same images for validation for now
        'names': {i: name for i, name in enumerate(dataset_config['labels']['classes'])}
    }
    
    with open(os.path.join(dataset_path, 'dataset.yaml'), 'w') as f:
        yaml.dump(dataset_yaml, f)
    
    logger.info(f"Dataset prepared with {processed_images} images")
    return dataset_path