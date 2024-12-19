from app import celery
from celery.utils.log import get_task_logger
import subprocess
import os
from bson import ObjectId
import datetime
from flask import current_app

logger = get_task_logger(__name__)


@celery.task(bind=True)
def train_model(self, current_user_id, folder_id, training_job_id, dataset_yaml_path, epochs):
    """Background task to train the model"""
    try:
        # Update initial status
        current_app.mongo.db.training_jobs.update_one(
            {'training_job_id': training_job_id},
            {'$set': {
                'status': 'training',
                'progress': 0,
                'updated_at': datetime.datetime.utcnow()
            }}
        )

        train_command = [
            'cmd', '/c',
            'D:\\Pribadi\\Sidehustle\\p24_v2\\venv\\Scripts\\activate', '&&',
            'yolo',
            'task=detect',
            'mode=train',
            'model=yolov8s.pt',
            f'data={dataset_yaml_path}',
            f'epochs={epochs}',
            f'project={current_user_id}',
            f'name={folder_id}',
        ]

        model_path = f'D:/Pribadi/Sidehustle/p24_v2/backend_flask/{current_user_id}/{folder_id}'

        # Create process with pipe for output
        process = subprocess.Popen(
            train_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Monitor training progress
        for line in iter(process.stdout.readline, ''):
            if 'epoch' in line.lower():
                try:
                    current_epoch = int(line.split('epoch')[1].split('/')[0])
                    progress = int((current_epoch / epochs) * 100)
                    
                    self.update_state(state='PROGRESS', meta={'progress': progress})
                    current_app.mongo.db.training_jobs.update_one(
                        {'training_job_id': training_job_id},
                        {'$set': {
                            'progress': progress,
                            'updated_at': datetime.datetime.utcnow()
                        }}
                    )
                except Exception as e:
                    logger.error(f"Error parsing progress: {e}")

        process.wait()

        if process.returncode == 0:
            best_model_path = os.path.join(model_path, 'weights', 'best.pt')
            
            # Save to GridFS
            with open(best_model_path, 'rb') as f:
                fs_id = current_app.fs.put(f, filename='best.pt', user_id=current_user_id)

            # Store model reference
            model_record = {
                'user_id': current_user_id,
                'folder_id': folder_id,
                'training_job_id': training_job_id,
                'model_path': best_model_path,
                'fs_id': fs_id,
                'created_at': datetime.datetime.utcnow()
            }
            current_app.mongo.db.trained_models.insert_one(model_record)

            # Update job status
            current_app.mongo.db.training_jobs.update_one(
                {'training_job_id': training_job_id},
                {'$set': {
                    'status': 'completed',
                    'progress': 100,
                    'model_path': best_model_path,
                    'updated_at': datetime.datetime.utcnow()
                }}
            )

            return {'status': 'completed', 'model_path': best_model_path}
        else:
            raise subprocess.CalledProcessError(process.returncode, train_command)

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        current_app.mongo.db.training_jobs.update_one(
            {'training_job_id': training_job_id},
            {'$set': {
                'status': 'failed',
                'progress': 0,
                'error': str(e),
                'updated_at': datetime.datetime.utcnow()
            }}
        )
        raise e