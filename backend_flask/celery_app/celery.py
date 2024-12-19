from celery import Celery
from pymongo import MongoClient
import gridfs

# MongoDB Configuration
MONGO_URI = 'mongodb://localhost:27017/mlweb_v2'

# Create MongoDB client
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client.get_database()

# Create GridFS
fs = gridfs.GridFS(mongo_db)

# Create Celery app
app = Celery('yolo_training',
             broker='redis://localhost:6379/0',
             backend='redis://localhost:6379/0',
             include=['celery_app.tasks'])

app.conf.update(
    result_backend='redis://localhost:6379/0',
    task_track_started=True,
    task_time_limit=18000,
    worker_max_memory_per_child=4000000
)

# Make mongo and fs available to tasks
app.mongo_db = mongo_db
app.fs = fs