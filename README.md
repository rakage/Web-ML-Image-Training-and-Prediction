# ML Web Application

A machine learning web application built with Flask, Django, and Celery for distributed task processing.

## System Architecture

The system consists of multiple components:
- Flask backend for API services
- Django web application for the main interface
- Celery for asynchronous task processing
- Redis for message broker and result backend
- Machine learning models for predictions

## Prerequisites

- Python 3.8+
- Redis
- Docker
- pip

## Installation

1. Clone the repository
```bash
git clone <repository-url>
cd <project-directory>
```

2. Install Python dependencies
```bash
pip install -r requirements.txt
```

3. Set up Redis using Docker
```bash
docker pull redis
docker run --name redis -d -p 6379:6379 redis
```

## Running the Application

### Flask Backend
1. Navigate to the Flask backend directory
```bash
cd backend_flask
```

2. Start the Flask server
```bash
python app.py
```

### Django Web Application
1. Navigate to the Django project directory
```bash
cd ml_web
```

2. Start the Django development server
```bash
python manage.py runserver
```

### Celery Worker
Start the Celery worker for processing background tasks:
```bash
celery -A app.celery worker --pool=solo --loglevel=info
```

## Project Structure

```
├── backend_flask/
│   ├── app.py
│   └── ...
├── ml_web/
│   ├── manage.py
│   └── ...
├── requirements.txt
└── README.md
```

## Features

1. Real-time Machine Learning Predictions
2. Distributed Task Processing
3. RESTful API Services
4. Web Interface for Model Management
5. Asynchronous Processing with Celery
6. Redis-based Caching and Message Broker

## Contributing

Please read our contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


![image](https://github.com/user-attachments/assets/cda243e7-2776-4e19-aee4-87e73b1640b2)

![image](https://github.com/user-attachments/assets/a6ea13de-9b6b-478b-8662-9996011cc6d1)

![image](https://github.com/user-attachments/assets/d724fa96-5729-4a6b-a9c0-99c123bff49e)

![image](https://github.com/user-attachments/assets/22aed81e-3ae9-42f7-bb6c-33fef14c2134)

![image](https://github.com/user-attachments/assets/3e87c32b-e175-4812-8896-ee91492c4772)

![image](https://github.com/user-attachments/assets/2237330a-45cc-4e62-80c1-786ba3108c44)
