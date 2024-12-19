from flask import Flask, request, jsonify, make_response, Response
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from bson import ObjectId
import datetime
import os
import json
from flask_cors import CORS
from io import BytesIO
import gridfs
from ultralytics import YOLO
import shutil
import cv2
import uuid
import yaml
import io
import numpy as np
# from celery_app.tasks import train_yolo_model
import torch
from torchvision.io import decode_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, retinanet_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, RetinaNet_ResNet50_FPN_Weights
from torchvision.models import resnet50, mobilenet_v3_large
from torchvision.models import ResNet50_Weights, MobileNet_V3_Large_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.io.image import read_image
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from PIL import Image
import io
import base64
import subprocess
from threading import Thread
import random
from celery import Celery
from celery_config import make_celery
from celery.utils.log import get_task_logger
import platform

# Initialize Celery
app = Flask(__name__)
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379/0',
    CELERY_RESULT_BACKEND='redis://localhost:6379/0'
)
celery = Celery(
    app.name,
    broker=app.config['CELERY_BROKER_URL'],
    backend=app.config['CELERY_RESULT_BACKEND']
)
celery.conf.update(
    broker_connection_retry_on_startup=True,
    worker_pool_restarts=True
)

if platform.system() == 'Windows':
    celery.conf.update(
        broker_connection_retry=True,
        broker_connection_max_retries=10,
        task_track_started=True,
        worker_pool='solo',  # Use solo pool for Windows
        worker_max_tasks_per_child=1
    )

class ContextTask(celery.Task):
    def __call__(self, *args, **kwargs):
        with app.app_context():
            return self.run(*args, **kwargs)

celery.Task = ContextTask

# Set up logger
logger = get_task_logger(__name__)

# Configuration
app.config['MONGO_URI'] = 'mongodb://localhost:27017/mlweb_v2'
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'caowfjca9812321kfafqw')  # Use environment variable
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = datetime.timedelta(hours=1)

# Initialize extensions
mongo = PyMongo(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# Initialize GridFS for storing images
fs = gridfs.GridFS(mongo.db)

CORS(app, origins=["http://localhost:8000", "http://127.0.0.1:8000"], supports_credentials=True)

# Load object detection models and weights
weights_fasterrcnn = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
weights_retinanet = RetinaNet_ResNet50_FPN_Weights.DEFAULT

model_fasterrcnn = fasterrcnn_resnet50_fpn_v2(weights=weights_fasterrcnn, box_score_thresh=0.9)
model_retinanet = retinanet_resnet50_fpn(weights=weights_retinanet, box_score_thresh=0.9)

# Load classification models and weights
weights_resnet = ResNet50_Weights.DEFAULT
weights_mobilenet = MobileNet_V3_Large_Weights.DEFAULT

model_resnet = resnet50(weights=weights_resnet)
model_mobilenet = mobilenet_v3_large(weights=weights_mobilenet)

# Set models to evaluation mode
model_fasterrcnn.eval()
model_retinanet.eval()
model_resnet.eval()
model_mobilenet.eval()

# Initialize preprocessing transforms
preprocess_fasterrcnn = weights_fasterrcnn.transforms()
preprocess_retinanet = weights_retinanet.transforms()

# Classification preprocessing
preprocess_classification = Compose([
    Resize(256),
    CenterCrop(224),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom JSON encoder to handle ObjectId
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)
app.json_encoder = JSONEncoder



# User Registration
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"msg": "Missing username or password"}), 400

    existing_user = mongo.db.users.find_one({'username': username})
    if existing_user:
        return jsonify({"msg": "Username already exists"}), 409

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

    mongo.db.users.insert_one({
        'username': username,
        'password': hashed_password
    })

    return jsonify({"msg": "User created successfully"}), 201

# User Login
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = mongo.db.users.find_one({'username': username})
    
    if user and bcrypt.check_password_hash(user['password'], password):
        access_token = create_access_token(identity=str(user['_id']))
        return jsonify(access_token=access_token), 200
    
    return jsonify({"msg": "Invalid credentials"}), 401

# User Profile
@app.route('/profile', methods=['GET'])
@jwt_required()
def profile():
    current_user_id = get_jwt_identity()
    user = mongo.db.users.find_one({'_id': ObjectId(current_user_id)}, {'password': 0})
    
    if not user:
        return jsonify({"msg": "User not found"}), 404
    
    return jsonify(user), 200

# User Logout
@app.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    return jsonify({"msg": "Successfully logged out"}), 200

# Upload Image (to MongoDB)
# @app.route('/upload-image', methods=['POST'])
# @jwt_required()
# def upload_image():
#     current_user_id = get_jwt_identity()
#     user = mongo.db.users.find_one({'_id': ObjectId(current_user_id)})
    
#     if not user:
#         return jsonify({"msg": "User not found"}), 404
    
#     if 'image' not in request.files:
#         return jsonify({"msg": "No image file found in the request"}), 400
    
#     # Get image file from request
#     image_file = request.files.get('image')
    
#     if not image_file:
#         return jsonify({"msg": "No image file found in the request"}), 400
    
#     # Process the image
#     img = Image.open(image_file)
#     img_byte_arr = BytesIO()
#     img.save(img_byte_arr, format='PNG')
#     img_byte_arr.seek(0)  # Reset the pointer to the start of the image byte array
    
#     # Save image to GridFS (MongoDB storage)
#     fs_id = fs.put(img_byte_arr, filename=image_file.filename, user_id=current_user_id)
    
#     # Save the image metadata to MongoDB
#     image_data = {
#         'user_id': current_user_id,
#         'image_id': fs_id,
#         'filename': image_file.filename,
#         'upload_date': datetime.datetime.utcnow()
#     }
#     mongo.db.images.insert_one(image_data)
    
#     return jsonify({"msg": "Image uploaded successfully", "image_id": str(fs_id)}), 201

@app.route('/upload-image', methods=['POST'])
@jwt_required()
def upload_image():
    current_user_id = get_jwt_identity()
    user = mongo.db.users.find_one({'_id': ObjectId(current_user_id)})

    if not user:
        return jsonify({"msg": "User not found"}), 404

    if 'image' not in request.files:
        return jsonify({"msg": "No image file found in the request"}), 400

    # Get image file from request
    image_file = request.files.get('image')
    folder_id = request.form.get('folder_id')  # Get the folder ID from form data

    if not image_file:
        return jsonify({"msg": "No image file found in the request"}), 400

    # Check if the folder ID exists
    if folder_id:
        folder = mongo.db.folders.find_one({'_id': ObjectId(folder_id), 'user_id': current_user_id})
        if not folder:
            print('here')
            return jsonify({"msg": "Folder not found"}), 404
    else:
        print('here2')
        folder_id = None  # If no folder is provided, set to None

    # Process the image
    img = Image.open(image_file)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)  # Reset the pointer to the start of the image byte array

    # Save image to GridFS (MongoDB storage)
    fs_id = fs.put(img_byte_arr, filename=image_file.filename, user_id=current_user_id)

    # Save the image metadata to MongoDB, including folder_id
    image_data = {
        'user_id': current_user_id,
        'image_id': fs_id,
        'filename': image_file.filename,
        'upload_date': datetime.datetime.utcnow()
    }
    result = mongo.db.images.insert_one(image_data)
    img_id = result.inserted_id

    # If folder_id is provided, add image to folder
    if folder_id:
        mongo.db.folders.update_one(
            {'_id': ObjectId(folder_id)},
            {'$push': {'image_list': img_id}}
        )

    return jsonify({"msg": "Image uploaded successfully", "image_id": str(img_id)}), 201


# Get Images for a User
@app.route('/images', methods=['GET'])
@jwt_required()
def get_images():
    current_user_id = get_jwt_identity()

    # Find all images for the current user
    images = mongo.db.images.find({'user_id': current_user_id})
    
    image_list = []
    for image in images:
        image_data = {
            'image_id': str(image['_id']),
            'filename': image['filename'],
            'upload_date': image['upload_date'],
            'image_url': f'/image/{str(image["_id"])}'  # Add image URL here
        }
        image_list.append(image_data)
    
    return jsonify(image_list), 200


@app.route('/image/<image_id>', methods=['GET'])
@jwt_required()
def get_image(image_id):
    current_user_id = get_jwt_identity()

    # Find the image in GridFS
    image_data = mongo.db.images.find_one({'_id': ObjectId(image_id), 'user_id': current_user_id})
    
    if not image_data:
        return jsonify({"msg": "Image not found"}), 404

    # Fetch the image from GridFS
    file = fs.get(ObjectId(image_data['image_id']))
    
    # Return the image file as a response
    return Response(file, content_type='image/png')

# Delete Image
@app.route('/image/<image_id>', methods=['DELETE'])
@jwt_required()
def delete_image(image_id):
    current_user_id = get_jwt_identity()
    
    # Find the image
    image = mongo.db.images.find_one({'_id': ObjectId(image_id), 'user_id': current_user_id})
    if not image:
        return jsonify({"msg": "Image not found"}), 404
    
    # Delete the image from GridFS
    fs.delete(ObjectId(image['image_id']))
    
    # Delete the image metadata
    mongo.db.images.delete_one({'_id': ObjectId(image_id)})

    # Remove the image from all folders
    mongo.db.folders.update_many(
        {'user_id': current_user_id},
        {'$pull': {'image_list': ObjectId(image_id)}}
    )
    
    return jsonify({"msg": "Image deleted successfully"}), 200

# Create Folder for User's Images
@app.route('/create-folder', methods=['POST'])
@jwt_required()
def create_folder():
    current_user_id = get_jwt_identity()
    
    data = request.get_json()
    folder_name = data.get('folder_name')
    
    if not folder_name:
        return jsonify({"msg": "Folder name is required"}), 400
    
    # Create folder in MongoDB (you can store metadata here)
    folder_data = {
        'user_id': current_user_id,
        'folder_name': folder_name,
        'created_at': datetime.datetime.utcnow()
    }
    folder = mongo.db.folders.insert_one(folder_data)
    
    return jsonify({"msg": "Folder created successfully", "folder_id": str(folder.inserted_id)}), 201

@app.route('/add-to-folder', methods=['POST'])
@jwt_required()
def add_to_folder():
    current_user_id = get_jwt_identity()
    
    data = request.get_json()
    image_ids = data.get('image_ids')  # List of image IDs
    folder_id = data.get('folder_id')
    
    if not image_ids or not folder_id:
        return jsonify({"msg": "Image IDs and Folder ID are required"}), 400
    
    if not isinstance(image_ids, list):
        return jsonify({"msg": "Image IDs must be a list"}), 400
    
    # Check if folder exists
    folder = mongo.db.folders.find_one({'_id': ObjectId(folder_id), 'user_id': current_user_id})
    if not folder:
        return jsonify({"msg": "Folder not found"}), 404
    
    # Initialize a list to keep track of failed images
    failed_images = []
    
    for image_id in image_ids:
        # Check if the image exists
        image = mongo.db.images.find_one({'_id': ObjectId(image_id), 'user_id': current_user_id})
        if not image:
            failed_images.append({"image_id": image_id, "error": "Image not found"})
            continue  # Skip this image
        
        # Check if the image is already in the folder (in image_list)
        if ObjectId(image_id) in folder.get('image_list', []):
            failed_images.append({"image_id": image_id, "error": "Image is already in the folder"})
            continue  # Skip this image
        
        # Add image ID to the folder's image_list
        mongo.db.folders.update_one(
            {'_id': ObjectId(folder_id)},
            {'$push': {'image_list': ObjectId(image_id)}}
        )
    
    # If some images failed, return details
    if failed_images:
        return jsonify({
            "msg": "Some images could not be added to the folder",
            "failed_images": failed_images
        }), 400
    
    return jsonify({"msg": "Images added to folder successfully"}), 200


@app.route('/folders', methods=['GET'])
@jwt_required()
def get_folders():
    current_user_id = get_jwt_identity()
    
    # Find all folders for the current user
    folders = mongo.db.folders.find({'user_id': current_user_id})
    
    folder_list = []
    for folder in folders:
        folder_data = {
            'folder_id': str(folder['_id']),
            'folder_name': folder['folder_name'],
            'created_at': folder['created_at'],
            'folder_url': f'/folder/{str(folder["_id"])}',
            'image_count': len(folder.get('image_list', [])),
            'image_ids': [str(image_id) for image_id in folder.get('image_list', [])]
        }
        folder_list.append(folder_data)
    
    return jsonify(folder_list), 200

# Get Images in a Folder
@app.route('/folder/<folder_id>', methods=['GET'])
@jwt_required()
def get_folder_images(folder_id):
    current_user_id = get_jwt_identity()
    
    # Find the folder
    folder = mongo.db.folders.find_one({'_id': ObjectId(folder_id), 'user_id': current_user_id})
    if not folder:
        return jsonify({"msg": "Folder not found"}), 404
    
    image_details = []
    for image_id in folder.get('image_list', []):
        image = mongo.db.images.find_one({'_id': image_id})
        image_details.append({
            'image_id': str(image['_id']),
            'filename': image['filename'],
            'upload_date': image['upload_date'],
            'image_url': f'/image/{str(image["_id"])}'
        })

    folder_data = {
        'folder_id': str(folder['_id']),
        'folder_name': folder['folder_name'],
        'created_at': folder['created_at'],
        'image_count': len(folder.get('image_list', [])),
        'images': image_details
    }

    return jsonify(folder_data), 200

# Delete Image from Folder
@app.route('/folder/<folder_id>/remove-image', methods=['POST'])
@jwt_required()
def remove_image_from_folder(folder_id):
    current_user_id = get_jwt_identity()
    
    data = request.get_json()
    image_id = data.get('image_id')
    
    if not image_id:
        return jsonify({"msg": "Image ID is required"}), 400
    
    # Find the folder
    folder = mongo.db.folders.find_one({'_id': ObjectId(folder_id), 'user_id': current_user_id})
    if not folder:
        return jsonify({"msg": "Folder not found"}), 404
    
    # Check if the image exists in the folder
    if ObjectId(image_id) not in folder.get('image_list', []):
        return jsonify({"msg": "Image not found in the folder"}), 404
    
    # Remove the image from the folder
    mongo.db.folders.update_one(
        {'_id': ObjectId(folder_id)},
        {'$pull': {'image_list': ObjectId(image_id)}}
    )
    
    return jsonify({"msg": "Image removed from folder successfully"}), 200

def predict_object_detection(img, model, preprocess, categories):
    # Step 1: Apply preprocessing transforms
    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension

    # Step 2: Perform inference
    with torch.no_grad():
        prediction = model(img_tensor)

    # Step 3: Extract labels, bounding boxes, and scores
    labels = prediction[0]['labels']  # List of predicted class indices
    boxes = prediction[0]['boxes']  # Bounding boxes in [x1, y1, x2, y2]
    scores = prediction[0]['scores']  # Confidence scores for each prediction

    # Map label indices to category names
    label_names = [categories[i.item()] for i in labels]

    return boxes, label_names, scores

def predict_classification(img, model, preprocess, categories):
    # Apply preprocessing transforms
    img_tensor = preprocess(to_tensor(img)).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Get the top 5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    predictions = []
    for i in range(top5_prob.size(0)):
        predictions.append({
            "label": categories[top5_catid[i].item()],
            "score": float(top5_prob[i].item())
        })
    return predictions


def draw_result(img, boxes, labels):
    # Draw bounding boxes and labels on the image
    drawn_img = draw_bounding_boxes(img, boxes=boxes, labels=labels, colors="red", width=4, font_size=15)
    return to_pil_image(drawn_img)

@app.route('/list_models', methods=['GET'])
@jwt_required()
def list_models():
    current_user_id = get_jwt_identity()
    # List available models
    models = ["fasterrcnn", "retinanet","resnet", "mobilenet"]

    # Get the list of users trained models
    trained_models = mongo.db.trained_models.find({'user_id': current_user_id})
    name_list = []
    for model in trained_models:
        name_list.append(model['project_name'])

    # Append the models to the response
    models.extend(name_list)

    return jsonify(models), 200





@app.route('/predict', methods=['POST'])
@jwt_required()
def predict():
    
    current_user_id = get_jwt_identity()
    # Get the model type from the request form-data
    model_type = request.form.get("model", "fasterrcnn")

    # Get the image file from form-data
    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "No image file provided"}), 400

    # Read the image file
    image_bytes = image_file.read()

    try:
        # Load the image from the in-memory bytes using PIL
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Failed to process the image: {str(e)}"}), 400

    # Initialize response structure
    response = {}

    if model_type in ["fasterrcnn", "retinanet"]:
        # Object Detection
        if model_type == "fasterrcnn":
            model = model_fasterrcnn
            preprocess = preprocess_fasterrcnn
            categories = weights_fasterrcnn.meta["categories"]
        elif model_type == "retinanet":
            model = model_retinanet
            preprocess = preprocess_retinanet
            categories = weights_retinanet.meta["categories"]
        else:
            return jsonify({"error": "Invalid object detection model type"}), 400

        # Run object detection prediction
        boxes, label_names, scores = predict_object_detection(image, model, preprocess, categories)

        # Draw bounding boxes on the image
        drawn_img = draw_bounding_boxes(to_tensor(image), boxes=boxes, labels=label_names, colors="red", width=4)
        result_img = to_pil_image(drawn_img)

        # Convert the image to Base64
        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        # Add object detection results to response
        predictions = []
        for label, score, box in zip(label_names, scores, boxes):
            predictions.append({
                "label": label,
                "score": float(score.item()),
                "box": [float(coord) for coord in box.tolist()]
            })

        response["predictions"] = predictions
        response["image"] = img_base64

    elif model_type in ["resnet", "mobilenet"]:
        # Classification
        if model_type == "resnet":
            model = model_resnet
            preprocess = preprocess_classification
            categories = weights_resnet.meta["categories"]
        elif model_type == "mobilenet":
            model = model_mobilenet
            preprocess = preprocess_classification
            categories = weights_mobilenet.meta["categories"]
        else:
            return jsonify({"error": "Invalid classification model type"}), 400

        # Run classification prediction
        predictions = predict_classification(image, model, preprocess, categories)

        # Add classification results to response
        response["predictions"] = predictions

    else:
        # Get custom models
        trained_model = mongo.db.trained_models.find_one({'user_id': current_user_id, 'project_name': model_type})
        if not trained_model:
            return jsonify({"error": "Model not found"}), 404
        
        # Load the custom model
        model_path = trained_model['model_path']
        model = YOLO(model_path)

        # Perform object detection
        results = model(image)

        # Draw bounding boxes on the image
        if isinstance(image, Image.Image):  # If the input image is a PIL image
            result_img = np.array(image)  # Convert PIL to NumPy array
        else:
            result_img = image.copy()
        probs = []
        kotak = []
        predictions = []
        threshold = 0.7
        for i, r in enumerate(results):
            # Extract the bounding boxes and confidence scores
            boxes = r.boxes.cpu().numpy()
            data = boxes.data  # The array with bounding box info
            # Filter bounding boxes based on the confidence threshold
            filtered_boxes = data[data[:, 4] > threshold]  # Filter rows where confidence > threshold

            cls = r.names
            # Draw the filtered bounding boxes
            for box in filtered_boxes:
                x1, y1, x2, y2, conf, cls_idx = box  # Unpack values from the box
                class_name = cls[int(cls_idx)]
                label = f"{conf:.2f}"  # Create label with confidence score

                # Draw the bounding box on the image
                result_img = cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                result_img = cv2.putText(result_img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.5, (0, 255, 0), 1)
                predictions.append({
                    "score": float(conf),
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "label": class_name
                })
            
        result_img_pil = Image.fromarray(result_img)  # Convert BGR to RGB
        # Convert the image to Base64
        img_byte_arr = io.BytesIO()
        result_img_pil.save(img_byte_arr, format='JPEG')

        # Save to local
        result_img_pil.save(f"{str(uuid.uuid4())}.jpg")
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        response["image"] = img_base64
        # probs_serializable = [p.tolist() for p in probs]
        # response["box"] = kotak
        # response["predictions"] = probs_serializable

        response["predictions"] = predictions

    return jsonify(response)


@app.route('/save-bboxes', methods=['GET', 'POST', 'DELETE'])
@jwt_required()
def save_bboxes():
    current_user_id = get_jwt_identity()

    if request.method == 'POST':
        data = request.get_json()
        if not data or 'image_id' not in data or 'bboxes' not in data:
            return jsonify({"msg": "Invalid request body"}), 400

        image_id = data['image_id']
        boxes = data['bboxes']

        # Validate that image belongs to current user
        image_data = mongo.db.images.find_one({'_id': ObjectId(image_id), 'user_id': current_user_id})
        if not image_data:
            return jsonify({"msg": "Image not found or does not belong to user"}), 404

        # Validate boxes array format
        # Expecting boxes to be a list of dicts with x, y, width, height
        # For example: [{"x": 10, "y": 20, "width": 100, "height": 120}, ...]
        for box in boxes:
            if not all(key in box for key in ["x", "y", "width", "height"]):
                return jsonify({"msg": "Invalid box format"}), 400
            
            if 'name' not in box:
                box['name'] = f"Box {boxes.index(box) + 1}"

        # Upsert the bounding box data into a separate collection (e.g., "bboxes")
        # If you want versioning, you can insert a new document each time instead of upsert.
        # For simplicity, we do an upsert to keep only the latest bounding boxes.
        bbox_document = {
            'user_id': current_user_id,
            'image_id': ObjectId(image_id),
            'boxes': boxes,
            'updated_at': datetime.datetime.utcnow()
        }

        # Try to find if a bbox doc already exists for this image and user
        existing_bbox = mongo.db.bboxes.find_one({'image_id': ObjectId(image_id), 'user_id': current_user_id})

        if existing_bbox:
            # Update existing document
            mongo.db.bboxes.update_one(
                {'_id': existing_bbox['_id']},
                {'$set': bbox_document}
            )
        else:
            # Insert new document
            mongo.db.bboxes.insert_one(bbox_document)

        return jsonify({"msg": "Bounding boxes saved successfully"}), 200
    
    elif request.method == 'GET':
        # Retrieve existing bounding boxes
        image_id = request.args.get('image_id')
        
        if not image_id:
            return jsonify({"msg": "No image ID provided"}), 400

        # Find bounding boxes for this image and user
        bbox_data = mongo.db.bboxes.find_one({
            'image_id': ObjectId(image_id), 
            'user_id': current_user_id
        })
        print(bbox_data)

        if bbox_data:
            return jsonify({
                "bboxes": bbox_data.get('boxes', [])
            }), 200
        
        return jsonify({"bboxes": []}), 200
    
    elif request.method == 'DELETE':
        # Delete bounding boxes for an image
        data = request.get_json()
        if not data or 'image_id' not in data:
            return jsonify({"msg": "Invalid request body"}), 400

        image_id = data['image_id']

        # Validate that image belongs to current user
        image_data = mongo.db.images.find_one({'_id': ObjectId(image_id), 'user_id': current_user_id})
        if not image_data:
            return jsonify({"msg": "Image not found or does not belong to user"}), 404

        # Delete the bounding box data
        mongo.db.bboxes.delete_one({'image_id': ObjectId(image_id), 'user_id': current_user_id})

        return jsonify({"msg": "Bounding boxes deleted successfully"}), 200

# def train_model(images_dir, labels_dir, training_job_id, user_id, folder_id, folder_name, epochs, batch_size):
#     """
#     Train a YOLOv5 object detection model
    
#     Args:
#         images_dir (str): Directory containing training images
#         labels_dir (str): Directory containing label files
#         training_job_id (str): Unique identifier for the training job
#         user_id (str): ID of the user initiating the training
#         folder_id (str): ID of the folder containing training data
#         folder_name (str): Name of the folder containing training data
#         epochs (int): Number of training epochs
#         batch_size (int): Training batch size
#     """
#     try:
#         # Update job status to training
#         mongo.db.training_jobs.update_one(
#             {'_id': training_job_id},
#             {'$set': {
#                 'status': 'training',
#                 'progress': 50,
#                 'updated_at': datetime.datetime.utcnow()
#             }}
#         )
#         # class_name = list(set(folder_name))
#         # Prepare training directories
#         base_train_dir = f'../user_models/{user_id}/{folder_id}'
#         os.makedirs(base_train_dir, exist_ok=True)
        
#         flattened = [item for sublist in folder_name for item in sublist]
#         class_name = list(set(flattened))
#         # Prepare YOLO dataset configuration
#         dataset_config = {
#             'path': base_train_dir,
#             'train': 'images',
#             'val': 'images',
#             'nc': len(class_name),
#             'names': class_name
#         }

#         # Write dataset configuration
#         dataset_yaml_path = os.path.join(base_train_dir, 'dataset.yaml')
#         with open(dataset_yaml_path, 'w') as f:
#             yaml.dump(dataset_config, f)

#         # Prepare data structure for YOLOv5 training
#         train_images_dir = os.path.join(base_train_dir, 'images')
#         train_labels_dir = os.path.join(base_train_dir, 'labels')
#         os.makedirs(train_images_dir, exist_ok=True)
#         os.makedirs(train_labels_dir, exist_ok=True)

#         # Copy images and labels
#         for filename in os.listdir(images_dir):
#             shutil.copy(
#                 os.path.join(images_dir, filename), 
#                 os.path.join(train_images_dir, filename)
#             )
        
#         for filename in os.listdir(labels_dir):
#             shutil.copy(
#                 os.path.join(labels_dir, filename), 
#                 os.path.join(train_labels_dir, filename)
#             )

#         # Prepare training arguments
#         model_path = os.path.join(base_train_dir, 'trained_model')
        
#         # Determine appropriate device
#         device = '0' if torch.cuda.is_available() else 'cpu'

#         # Construct training command
#         train_command = [
#             'cmd', '/c',
#             'D:\\Pribadi\\Sidehustle\\p24_v2\\venv\\Scripts\\activate', '&&',
#             'python', 'yolov5/train.py',
#             '--img', '640',
#             '--batch', str(batch_size),
#             '--epochs', str(epochs),
#             '--data', dataset_yaml_path,
#             '--weights', 'yolov5s.pt',  # Start with pre-trained weights
#             '--device', device,
#             '--project', model_path
#         ]

#         # Run training
#         try:
#             # Update job status to training in progress
#             mongo.db.training_jobs.update_one(
#                 {'_id': training_job_id},
#                 {'$set': {
#                     'status': 'training',
#                     'progress': 75,
#                     'updated_at': datetime.datetime.utcnow()
#                 }}
#             )

#             # Execute training
#             result = subprocess.run(
#                 train_command, 
#                 capture_output=True, 
#                 text=True, 
#                 check=True
#             )

#             # Find the best model
#             best_model_path = os.path.join(model_path, 'weights', 'best.pt')
            
#             # Store model reference in database
#             model_record = {
#                 'user_id': user_id,
#                 'folder_id': folder_id,
#                 'training_job_id': training_job_id,
#                 'model_path': best_model_path,
#                 'created_at': datetime.datetime.utcnow(),
#                 'training_logs': result.stdout
#             }
#             mongo.db.trained_models.insert_one(model_record)

#             # Update job status to completed
#             mongo.db.training_jobs.update_one(
#                 {'_id': training_job_id},
#                 {'$set': {
#                     'status': 'completed',
#                     'progress': 100,
#                     'model_path': best_model_path,
#                     'updated_at': datetime.utcnow()
#                 }}
#             )

#             # Save to GridFS
#             with open(best_model_path, 'rb') as f:
#                 fs_id = fs.put(f, filename='best.pt', user_id=user_id)

#             # Update model record with GridFS ID
#             mongo.db.trained_models.update_one(
#                 {'_id': model_record['_id']},
#                 {'$set': {'fs_id': fs_id}}
#             )

#             return best_model_path

#         except subprocess.CalledProcessError as train_error:
#             # Log training error
#             error_details = {
#                 'stdout': train_error.stdout,
#                 'stderr': train_error.stderr,
#                 'return_code': train_error.returncode
#             }
            
#             # Update job status to failed
#             mongo.db.training_jobs.update_one(
#                 {'_id': training_job_id},
#                 {'$set': {
#                     'status': 'failed',
#                     'progress': 0,
#                     'error': str(error_details),
#                     'updated_at': datetime.datetime.utcnow()
#                 }}
#             )
            
#             raise RuntimeError(f"Training failed: {error_details}")

#     except Exception as e:
#         # Catch any unexpected errors
#         mongo.db.training_jobs.update_one(
#             {'_id': training_job_id},
#             {'$set': {
#                 'status': 'failed',
#                 'progress': 0,
#                 'error': str(e),
#                 'updated_at': datetime.datetime.utcnow()
#             }}
#         )
#         raise

# @app.route('/train', methods=['POST'])
# @jwt_required()
# def start_training():
#     current_user_id = get_jwt_identity()
#     json_data = request.get_json()
#     folder_id = json_data.get('folder_id')
#     epochs = json_data.get('epochs', 10)
#     batch_size = json_data.get('batch_size', 4)

#     # Find the folder and its images
#     folder = mongo.db.folders.find_one({'_id': ObjectId(folder_id), 'user_id': current_user_id})
#     if not folder:
#         return jsonify({"msg": "Folder not found"}), 404

#     # Check if there are enough images to train
#     images_count = len(folder.get('image_list', []))
#     if images_count < 5:  # Minimum number of images required for training
#         return jsonify({"msg": "Not enough images to start training. Minimum 5 images required."}), 400

#     # Generate a unique training job ID
#     training_job_id = str(uuid.uuid4())

#     # Prepare training data
#     train_folder = f'training_data/{current_user_id}/{folder_id}'
#     folder_name = folder.get('folder_name', 'unnamed_folder')
#     os.makedirs(train_folder, exist_ok=True)
#     images_dir = os.path.join(train_folder, "images")
#     labels_dir = os.path.join(train_folder, "labels")
#     os.makedirs(images_dir, exist_ok=True)
#     os.makedirs(labels_dir, exist_ok=True)

#     # Prepare the training data
#     def prepare_training_data():
#         try:
#             # Store training job details in database
#             training_job = {
#                 '_id': training_job_id,
#                 'user_id': current_user_id,
#                 'folder_id': folder_id,
#                 'status': 'preparing',
#                 'progress': 0,
#                 'created_at': datetime.datetime.utcnow(),
#                 'updated_at': datetime.datetime.utcnow()
#             }
#             mongo.db.training_jobs.insert_one(training_job)

#             # Prepare images and labels
#             prepared_images = []
#             classes = []
#             for image_id in folder.get('image_list', []):
#                 image_data = mongo.db.images.find_one({'_id': image_id})
                
#                 unique_classes = list(set(bbox['name'] for image in folder.get('image_list', []) 
#                             for bbox_data in [mongo.db.bboxes.find_one({'image_id': ObjectId(image_id)})]
#                             if bbox_data and bbox_data.get('boxes')
#                             for bbox in bbox_data['boxes']))
                
#                 # Collect classes
#                 classes.append(unique_classes)
#                 class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
#                 if image_data:
#                     # Save image
#                     image_file = fs.get(ObjectId(image_data['image_id']))
#                     # Check if the image is labeled, if yes then save it
#                     bbox_data = mongo.db.bboxes.find_one({'image_id': ObjectId(image_data['_id'])})
#                     if not bbox_data or not bbox_data.get('boxes'):
#                         continue
                    
#                     image_path = os.path.join(images_dir, image_data['filename'])
#                     with open(image_path, 'wb') as f:
#                         f.write(image_file.read())

#                     # Open image to get dimensions
#                     img = Image.open(image_path)
#                     img_width, img_height = img.size

#                     # Prepare label file for this image
#                     label_path = os.path.join(labels_dir, f"{os.path.splitext(image_data['filename'])[0]}.txt")
                    
#                     with open(label_path, 'w') as label_file:
#                         # Iterate through all bounding boxes
#                         for bbox in bbox_data['boxes']:
#                             x = bbox['x']
#                             y = bbox['y']
#                             width = bbox['width']
#                             height = bbox['height']
                            
#                             # Get class index (you might want to map names to indices)
#                             class_name = bbox.get('name', 'unknown')
#                             class_index = class_to_index.get(class_name, 0)
                            
#                             # YOLO format
#                             center_x = (x + width / 2) / img_width
#                             center_y = (y + height / 2) / img_height
#                             w = width / img_width
#                             h = height / img_height

#                             # Write each bounding box to the label file
#                             label_file.write(f"{class_index} {center_x} {center_y} {w} {h}\n")
                            

#                     prepared_images.append(image_path)

#             # Update job status to preparing completed
#             mongo.db.training_jobs.update_one(
#                 {'_id': training_job_id},
#                 {'$set': {
#                     'status': 'prepared',
#                     'progress': 25,
#                     'updated_at': datetime.datetime.utcnow()
#                 }}
#             )

#             # Start training (you'll need to implement train_model function)
#             try:
#                 train_model(
#                     images_dir, 
#                     labels_dir, 
#                     training_job_id,
#                     current_user_id,
#                     folder_id,
#                     classes,
#                     epochs,
#                     batch_size
#                 )
#             except Exception as train_error:
#                 # Update job status to failed
#                 mongo.db.training_jobs.update_one(
#                     {'_id': training_job_id},
#                     {'$set': {
#                         'status': 'failed',
#                         'error': str(train_error),
#                         'updated_at': datetime.datetime.utcnow()
#                     }}
#                 )
#                 raise

#         except Exception as e:
#             # Update job status to failed
#             mongo.db.training_jobs.update_one(
#                 {'_id': training_job_id},
#                 {'$set': {
#                     'status': 'failed',
#                     'error': str(e),
#                     'updated_at': datetime.datetime.utcnow()
#                 }}
#             )
#             raise

#     # Start preparation and training in a background thread
#     thread = Thread(target=prepare_training_data)
#     thread.start()

#     # Immediately return the training job ID
#     return jsonify({
#         "msg": "Training initiated",
#         "training_job_id": training_job_id
#     }), 202
    
def prepare_yolo_dataset(folder, user_id, folder_id):
    # Base directory for the user's model
    base_train_dir = f'D:/Pribadi/Sidehustle/p24_v2/backend_flask/user_models/{user_id}/{folder_id}'
    os.makedirs(base_train_dir, exist_ok=True)

    # Create subdirectories
    splits = ['train', 'test', 'valid']
    for split in splits:
        split_images_dir = os.path.join(base_train_dir, split, 'images')
        split_labels_dir = os.path.join(base_train_dir, split, 'labels')
        os.makedirs(split_images_dir, exist_ok=True)
        os.makedirs(split_labels_dir, exist_ok=True)

    # Function to generate dynamic class mapping
    def generate_dynamic_class_mapping(folder_image_list):
        unique_classes = set()
        for image_id in folder_image_list:
            bbox_data = mongo.db.bboxes.find_one({'image_id': ObjectId(image_id)})
            if bbox_data and bbox_data.get('boxes'):
                unique_classes.update(
                    bbox.get('name', 'unknown') 
                    for bbox in bbox_data['boxes']
                )
        
        # Sort to ensure consistent mapping
        sorted_classes = sorted(list(unique_classes))
        class_to_index = {cls: idx for idx, cls in enumerate(sorted_classes)}
        
        return class_to_index, sorted_classes

    # Generate class mapping
    class_to_index, class_list = generate_dynamic_class_mapping(folder.get('image_list', []))

    # Prepare dataset configuration
    dataset_config = {
        'path': base_train_dir,
        'train': os.path.join(base_train_dir, 'train/images'),
        'val': os.path.join(base_train_dir, 'valid/images'),
        'test': os.path.join(base_train_dir, 'test/images'),
        'nc': len(class_list),
        'names': class_list
    }

    # Write dataset configuration
    dataset_yaml_path = os.path.join(base_train_dir, 'dataset.yaml')
    with open(dataset_yaml_path, 'w') as f:
        yaml.dump(dataset_config, f)

    # Prepare images and labels
    prepared_images = []
    
    # Split the images (you can adjust the split ratios as needed)
    image_list = folder.get('image_list', [])
    random.shuffle(image_list)
    
    train_split = int(len(image_list) * 0.7)
    test_split = train_split + int(len(image_list) * 0.15)
    
    for idx, image_id in enumerate(image_list):
        image_data = mongo.db.images.find_one({'_id': image_id})
        if image_data:
            # Determine which split this image belongs to
            if idx < train_split:
                split = 'train'
            elif idx < test_split:
                split = 'test'
            else:
                split = 'valid'
            
            # Paths for this split
            split_images_dir = os.path.join(base_train_dir, split, 'images')
            split_labels_dir = os.path.join(base_train_dir, split, 'labels')
            
            # Save image
            image_file = fs.get(ObjectId(image_data['image_id']))
            
            # Check if the image is labeled
            bbox_data = mongo.db.bboxes.find_one({'image_id': ObjectId(image_data['_id'])})
            if not bbox_data or not bbox_data.get('boxes'):
                continue
            
            # Image path
            image_path = os.path.join(split_images_dir, image_data['filename'])
            with open(image_path, 'wb') as f:
                f.write(image_file.read())

            # Open image to get dimensions
            img = Image.open(image_path)
            img_width, img_height = img.size

            # Prepare label file for this image
            label_path = os.path.join(split_labels_dir, f"{os.path.splitext(image_data['filename'])[0]}.txt")
            
            with open(label_path, 'w') as label_file:
                # Iterate through all bounding boxes
                for bbox in bbox_data['boxes']:
                    x = bbox['x']
                    y = bbox['y']
                    width = bbox['width']
                    height = bbox['height']
                    
                    # Get class index dynamically
                    class_name = bbox.get('name', 'unknown')
                    class_index = class_to_index.get(class_name, 0)
                    
                    # YOLO format
                    center_x = (x + width / 2) / img_width
                    center_y = (y + height / 2) / img_height
                    w = width / img_width
                    h = height / img_height

                    # Write each bounding box to the label file
                    label_file.write(f"{class_index} {center_x} {center_y} {w} {h}\n")

            prepared_images.append(image_path)
    
    return base_train_dir, dataset_yaml_path



# @app.route('/train', methods=['POST'])
# @jwt_required()
# def start_training():
#     current_user_id = get_jwt_identity()
#     json_data = request.get_json()
#     folder_id = json_data.get('folder_id')
#     epochs = json_data.get('epochs', 10)
#     batch_size = json_data.get('batch_size', 4)

#     # Find the folder and its images
#     folder = mongo.db.folders.find_one({'_id': ObjectId(folder_id), 'user_id': current_user_id})
#     if not folder:
#         return jsonify({"msg": "Folder not found"}), 404

#     # Check if there are enough images to train
#     images_count = len(folder.get('image_list', []))
#     if images_count < 5:  # Minimum number of images required for training
#         return jsonify({"msg": "Not enough images to start training. Minimum 5 images required."}), 400
    
#     # Generate a unique training job ID
#     training_job_id = str(uuid.uuid4())

#     training_record = {
#         'training_job_id': training_job_id,
#         'user_id': current_user_id,
#         'folder_id': folder_id,
#         'status': 'preparing',
#         'progress': 0,
#         'created_at': datetime.datetime.utcnow(),
#         'updated_at': datetime.datetime.utcnow()
#     }

#     # Store training job details in database
#     mongo.db.training_jobs.insert_one(training_record)

#     # Prepare training data
#     base_train_dir, dataset_yaml_path = prepare_yolo_dataset(folder, current_user_id, folder_id)

#     device = '0' if torch.cuda.is_available() else 'cpu'

#     train_command = [
#         'cmd', '/c',
#         'D:\\Pribadi\\Sidehustle\\p24_v2\\venv\\Scripts\\activate', '&&',
#         'yolo',
#         'task=detect',
#         'mode=train',
#         'model=yolov8s.pt',
#         f'data={dataset_yaml_path}',
#         f'epochs={epochs}',
#         f'project={current_user_id}',
#         f'name={folder_id}',
#     ]

#     model_path = f'D:/Pribadi/Sidehustle/p24_v2/backend_flask/{current_user_id}/{folder_id}'

#     # Run training
#     try:
#         # Update job status to training in progress
#         mongo.db.training_jobs.update_one(
#             {'training_job_id': training_job_id},
#             {'$set': {
#                 'status': 'training',
#                 'progress': 75,
#                 'updated_at': datetime.datetime.utcnow()
#             }}
#         )

#         # Execute training
#         result = subprocess.run(
#             train_command, 
#             capture_output=True, 
#             text=True, 
#             check=True
#         )

#         # Find the best model
#         best_model_path = os.path.join(model_path, 'weights', 'best.pt')
        
#         # Store model reference in database
#         model_record = {
#             'user_id': current_user_id,
#             'folder_id': folder_id,
#             'training_job_id': training_job_id,
#             'model_path': best_model_path,
#             'created_at': datetime.datetime.utcnow(),
#             'training_logs': result.stdout
#         }
#         mongo.db.trained_models.insert_one(model_record)

#         # Update job status to completed
#         mongo.db.training_jobs.update_one(
#             {'training_job_id': training_job_id},
#             {'$set': {
#                 'status': 'completed',
#                 'progress': 100,
#                 'model_path': best_model_path,
#                 'updated_at': datetime.datetime.utcnow()
#             }}
#         )

#         # Save to GridFS
#         with open(best_model_path, 'rb') as f:
#             fs_id = fs.put(f, filename='best.pt', user_id=current_user_id)

#         # Update model record with GridFS ID
#         mongo.db.trained_models.update_one(
#             {'_id': model_record['_id']},
#             {'$set': {'fs_id': fs_id}}
#         )

#         return jsonify({"msg": f"Training completed successfully. Model saved at {best_model_path}"}), 200
    
#     except subprocess.CalledProcessError as train_error:
#         # Log training error
#         error_details = {
#             'stdout': train_error.stdout,
#             'stderr': train_error.stderr,
#             'return_code': train_error.returncode
#         }
        
#         # Update job status to failed
#         mongo.db.training_jobs.update_one(
#             {'training_job_id': training_job_id},
#             {'$set': {
#                 'status': 'failed',
#                 'progress': 0,
#                 'error': str(error_details),
#                 'updated_at': datetime.datetime.utcnow()
#             }}
#         )

#         return jsonify({"msg": "Training failed", "error": error_details}), 500


@celery.task(bind=True)
def train_model(self, current_user_id, folder_id, project_name, training_job_id, dataset_yaml_path, epochs):
    """Background task to train the model with improved logging"""
    try:
        # Update initial status
        mongo.db.training_jobs.update_one(
            {'training_job_id': training_job_id},
            {'$set': {
                'status': 'training',
                'progress': 0,
                'logs': [],
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
            f'name={project_name}',
        ]

        model_path = f'D:/Pribadi/Sidehustle/p24_v2/backend_flask/{current_user_id}/{project_name}'

        # Use threading to handle output streams
        from threading import Thread
        from queue import Queue, Empty
        
        def enqueue_output(out, queue):
            for line in iter(out.readline, ''):
                queue.put(line)
            out.close()

        process = subprocess.Popen(
            train_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            shell=True,
            bufsize=1
        )

        # Create queues and threads for both stdout and stderr
        stdout_queue = Queue()
        stderr_queue = Queue()
        stdout_thread = Thread(target=enqueue_output, args=(process.stdout, stdout_queue))
        stderr_thread = Thread(target=enqueue_output, args=(process.stderr, stderr_queue))
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()

        # Initialize log storage
        training_logs = []
        current_epoch = 0
        import time

        # Process output with timeout
        while True:
            # Check if process has finished
            return_code = process.poll()
            if return_code is not None:
                break

            # Read from stdout queue
            try:
                while True:
                    line = stdout_queue.get_nowait().strip()
                    
                    # Store log
                    training_logs.append(line)
                    
                    # Update logs in database periodically
                    if len(training_logs) >= 10:
                        mongo.db.training_jobs.update_one(
                            {'training_job_id': training_job_id},
                            {'$push': {'logs': {'$each': training_logs}}}
                        )
                        training_logs = []

                    # Progress tracking
                    if 'Epoch' in line and 'GPU_mem' not in line:
                        try:
                            # Extract epoch number using regex
                            import re
                            current_values = line.split()
                            if len(current_values) > 1 and current_values[0].isdigit():
                                current_epoch = int(current_values[0])
                                progress = int((current_epoch / epochs) * 100)
                                
                                self.update_state(state='PROGRESS', meta={'progress': progress})
                                mongo.db.training_jobs.update_one(
                                    {'training_job_id': training_job_id},
                                    {'$set': {
                                        'progress': progress,
                                        'updated_at': datetime.datetime.utcnow()
                                    }}
                                )
                        except Exception as e:
                            logger.error(f"Error parsing progress: {e}")

            except Empty:
                pass

            # Read from stderr queue
            try:
                while True:
                    line = stderr_queue.get_nowait().strip()
                    training_logs.append(f"{line}")
            except Empty:
                pass

            # Small sleep to prevent CPU overuse
            time.sleep(0.1)

        # Update any remaining logs
        if training_logs:
            mongo.db.training_jobs.update_one(
                {'training_job_id': training_job_id},
                {'$push': {'logs': {'$each': training_logs}}}
            )

        # Process completion
        if process.returncode == 0:
            best_model_path = os.path.join(model_path, 'weights', 'best.pt')
            
            # Save to GridFS
            with open(best_model_path, 'rb') as f:
                fs_id = fs.put(f, filename='best.pt', user_id=current_user_id)

            # Store model reference
            model_record = {
                'user_id': current_user_id,
                'folder_id': folder_id,
                'project_name':project_name,
                'training_job_id': training_job_id,
                'model_path': best_model_path,
                'fs_id': fs_id,
                'created_at': datetime.datetime.utcnow()
            }
            mongo.db.trained_models.insert_one(model_record)

            # Update job status
            mongo.db.training_jobs.update_one(
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
            error_message = f"Process failed with return code {process.returncode}"
            raise subprocess.CalledProcessError(process.returncode, train_command, output=error_message)

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        mongo.db.training_jobs.update_one(
            {'training_job_id': training_job_id},
            {'$set': {
                'status': 'failed',
                'progress': 0,
                'error': str(e),
                'updated_at': datetime.datetime.utcnow()
            }}
        )
        raise e
    
@app.route('/train', methods=['POST'])
@jwt_required()
def start_training():
    current_user_id = get_jwt_identity()
    json_data = request.get_json()
    folder_id = json_data.get('folder_id')
    project_name = json_data.get('project_name')
    epochs = json_data.get('epochs', 10)
    batch_size = json_data.get('batch_size', 4)

    # Check project name exists
    if not project_name:
        return jsonify({"msg": "Project name is required"}), 400
    
    project_name_db = mongo.db.trained_models.find_one({'user_id': current_user_id, 'project_name': project_name})
    if project_name_db:
        return jsonify({"msg": "Project name already exists"}), 400

    # Find the folder and validate
    folder = mongo.db.folders.find_one({'_id': ObjectId(folder_id), 'user_id': current_user_id})
    if not folder:
        return jsonify({"msg": "Folder not found"}), 404

    if len(folder.get('image_list', [])) < 5:
        return jsonify({"msg": "Not enough images to start training. Minimum 5 images required."}), 400
    
    # Generate training job ID
    training_job_id = str(uuid.uuid4())

    # Create initial training record
    training_record = {
        'training_job_id': training_job_id,
        'user_id': current_user_id,
        'folder_id': folder_id,
        'project_name': project_name,
        'status': 'preparing',
        'progress': 0,
        'created_at': datetime.datetime.utcnow(),
        'updated_at': datetime.datetime.utcnow()
    }
    mongo.db.training_jobs.insert_one(training_record)

    # Prepare dataset
    base_train_dir, dataset_yaml_path = prepare_yolo_dataset(folder, current_user_id, folder_id)

    # Start background task
    task = train_model.delay(
        current_user_id,
        folder_id,
        project_name,
        training_job_id,
        dataset_yaml_path,
        epochs
    )

    return jsonify({
        "msg": "Training started",
        "training_job_id": training_job_id,
        "task_id": task.id
    }), 202

@app.route('/training-logs/<training_job_id>', methods=['GET'])
@jwt_required()
def get_training_logs(training_job_id):
    current_user_id = get_jwt_identity()
    
    # Find the training job
    training_job = mongo.db.training_jobs.find_one({
        'training_job_id': training_job_id,
        'user_id': current_user_id
    })
    
    if not training_job:
        return jsonify({"msg": "Training job not found"}), 404
        
    return jsonify({
        "status": training_job.get('status'),
        "progress": training_job.get('progress'),
        "logs": training_job.get('logs', [])
    }), 200

@app.route('/training-progress/<training_job_id>', methods=['GET'])
@jwt_required()
def get_training_progress(training_job_id):
    """Endpoint to check training progress"""
    current_user_id = get_jwt_identity()
    
    # Find training job
    training_job = mongo.db.training_jobs.find_one({
        'training_job_id': training_job_id,
        'user_id': current_user_id
    })
    
    if not training_job:
        return jsonify({"msg": "Training job not found"}), 404
        
    return jsonify({
        "status": training_job.get('status'),
        "progress": training_job.get('progress'),
        "error": training_job.get('error'),
        "model_path": training_job.get('model_path')
    }), 200

@app.route('/training-jobs', methods=['GET'])
@jwt_required()
def get_training_jobs():
    current_user_id = get_jwt_identity()

    # Find all training jobs for the current user
    training_jobs = mongo.db.training_jobs.find({'user_id': current_user_id})
    
    job_list = []
    for job in training_jobs:
        job_data = {
            'job_id': str(job['_id']),
            'status': job['status'],
            'progress': job['progress'],
            'created_at': job['created_at'],
            'updated_at': job['updated_at']
        }
        job_list.append(job_data)
    
    return jsonify(job_list), 200

@app.route('/training-job/<job_id>', methods=['GET'])
@jwt_required()
def get_training_job(job_id):
    current_user_id = get_jwt_identity()

    # Find the training job
    job = mongo.db.training_jobs.find_one({'_id': ObjectId(job_id), 'user_id': current_user_id})
    if not job:
        return jsonify({"msg": "Training job not found"}), 404

    return jsonify(job), 200

@app.route('/download-model/<model_id>', methods=['GET'])
@jwt_required()
def download_model(model_id):
    current_user_id = get_jwt_identity()

    # Find the model
    model = mongo.db.trained_models.find_one({'_id': ObjectId(model_id), 'user_id': current_user_id})
    if not model:
        return jsonify({"msg": "Model not found"}), 404

    # Get the model file from GridFS
    model_file = fs.get(ObjectId(model['model_path']))

    # Return the model file as a response
    return Response(model_file, content_type='application/octet-stream')


# Error Handlers
@jwt.unauthorized_loader
def unauthorized_response(callback):
    return make_response(jsonify({
        'msg': 'Missing Authorization Header'
    }), 401)

@jwt.invalid_token_loader
def invalid_token_response(callback):
    return make_response(jsonify({
        'msg': 'Invalid Token'
    }), 422)

if __name__ == '__main__':
    app.run(debug=True)
