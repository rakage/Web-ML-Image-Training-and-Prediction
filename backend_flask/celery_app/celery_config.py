from celery import Celery

celery = Celery('tasks', broker='redis://localhost:6379/0')
celery.conf.update(
    result_backend='redis://localhost:6379/0',
    task_track_started=True,
    task_time_limit=18000,  # 5 hours
    worker_max_memory_per_child=4000000  # 4GB
)