import os
import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet18
from flask import Flask, jsonify, request
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity
from pymongo import MongoClient
import gridfs
from bson import ObjectId
from flask_pymongo import PyMongo
from werkzeug.utils import secure_filename
from typing import List, Optional, Dict, Any
from PIL import Image
import io
import base64

# Configuration and Setup
app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'caowfjca9812321kfafqw'  # Replace with a secure secret key
app.config['MONGO_URI'] = 'mongodb://localhost:27017/mlweb_v2'
app.config['UPLOAD_FOLDER'] = r'D:\Pribadi\Sidehustle\p24_v2\backend_flask\models'  # Specify a directory
jwt = JWTManager(app)
mongo = PyMongo(app)
fs = gridfs.GridFS(mongo.db)

# Utility Functions
def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MultifolderMongoGridFSDataset(Dataset):
    def __init__(self, folder_ids: List[str], mongo_db, gridfs_db, transform=None):
        """
        Initialize dataset from multiple MongoDB GridFS image folders.
        
        :param folder_ids: List of folder ObjectIds
        :param mongo_db: MongoDB database connection
        :param gridfs_db: GridFS connection
        :param transform: Optional image transformations
        """
        self.mongo_db = mongo_db
        self.gridfs_db = gridfs_db
        self.transform = transform
        
        self.images = []
        self.labels = []
        self.label_map = {}
        
        # Collect images from all specified folders
        for folder_id in folder_ids:
            folder = mongo_db.folders.find_one({'_id': ObjectId(folder_id)})
            if not folder:
                print(f"Warning: Folder {folder_id} not found")
                continue
            
            for image_id in folder.get('image_list', []):
                image_doc = mongo_db.images.find_one({'_id': ObjectId(image_id)})
                if image_doc:
                    # Retrieve image from GridFS
                    try:
                        image_file = gridfs_db.get(ObjectId(image_doc['image_id']))
                        
                        # Convert to PIL Image
                        from PIL import Image
                        import io
                        image = Image.open(io.BytesIO(image_file.read()))
                        
                        # Manage labels and label mapping from folder name
                        label = folder.get('folder_name', 'unknown')
                        if label not in self.label_map:
                            self.label_map[label] = len(self.label_map)
                        
                        self.images.append(image)
                        self.labels.append(label)
                    except Exception as e:
                        print(f"Error processing image {image_id}: {e}")
        
        self.classes = sorted(self.label_map.keys())
        self.class_to_idx = self.label_map
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Convert label to index
        label_idx = self.class_to_idx[label]
        
        return image, label_idx

@app.route('/train', methods=['POST'])
# @jwt_required()
def train_model():
    """
    Flask endpoint to trigger the training process.
    Expects a JSON payload with 'folder_ids' to specify training folders.
    """
    try:
        # Parse input data
        data = request.get_json()
        folder_ids = data.get('folder_ids', [])
        num_epochs = data.get('epochs', 10)
        batch_size = data.get('batch_size', 32)
        learning_rate = data.get('learning_rate', 0.001)
        seed = data.get('seed', 42)
        
        # Set random seed for reproducibility
        set_seed(seed)
        
        # Dataset and DataLoader
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = MultifolderMongoGridFSDataset(folder_ids, mongo.db, fs, transform=transform)
        
        if len(dataset) == 0:
            return jsonify({'error': 'No images found for provided folder_ids.'}), 400
        
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Model, Loss, and Optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training Loop
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
            epoch_accuracy = correct / total * 100
            print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.2f}%")
        
        # Save the trained model
        model_save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'trained_model.pth')
        torch.save(model.state_dict(), model_save_path)
        
        # Save classes and class_to_idx in Mongo
        metadata = {
            'classes': dataset.classes,
            'class_to_idx': dataset.class_to_idx,
            'last_trained': time.time()
        }
        
        # Upsert metadata document (single record)
        mongo.db.model_metadata.update_one(
            {}, 
            {'$set': metadata}, 
            upsert=True
        )
        
        return jsonify({
            'message': 'Model training completed successfully',
            'model_path': model_save_path,
            'classes': dataset.classes
        })
    
    except Exception as e:
        print(f"Error during training: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
# @jwt_required()
def predict_image():
    """
    Flask endpoint to predict the class of an uploaded image.
    Expects a POST request with an image file named 'image'.
    """
    try:
        # Load the trained model
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'trained_model.pth')
        if not os.path.exists(model_path):
            return jsonify({'error': 'Trained model not found. Please train the model first.'}), 400
        
        # Retrieve class metadata from Mongo
        metadata = mongo.db.model_metadata.find_one({})
        if not metadata or 'classes' not in metadata or 'class_to_idx' not in metadata:
            return jsonify({'error': 'Model metadata not found. Please train the model first.'}), 400
        
        classes = metadata['classes']
        class_to_idx = metadata['class_to_idx']
        
        # Initialize the model with the correct number of output classes
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # Check if an image file is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided.'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected image file.'}), 400
        
        # Preprocess the image
        image = Image.open(file.stream).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0)
        
        # Predict the class
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1).numpy()[0]
            predicted_index = np.argmax(probabilities)
            predicted_class = classes[predicted_index]
            confidence = float(probabilities[predicted_index])
        
        # Build a probability dictionary for all classes
        class_probabilities = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': class_probabilities
        })
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
