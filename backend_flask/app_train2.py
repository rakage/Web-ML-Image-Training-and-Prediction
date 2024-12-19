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
app.config['UPLOAD_FOLDER'] = 'backend_flask\models'  # Specify a temporary upload directory
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

# Custom Dataset for MongoDB-stored Images
class MongoGridFSDataset(Dataset):
    def __init__(self, folder_id, mongo_db, gridfs_db, transform=None):
        """
        Initialize dataset from MongoDB GridFS images in a specific folder.
        
        :param folder_id: ObjectId of the folder
        :param mongo_db: MongoDB database connection
        :param gridfs_db: GridFS connection
        :param transform: Optional image transformations
        """
        self.mongo_db = mongo_db
        self.gridfs_db = gridfs_db
        self.transform = transform
        
        # Fetch folder details
        folder = mongo_db.folders.find_one({'_id': ObjectId(folder_id)})
        if not folder:
            raise ValueError(f"Folder {folder_id} not found")
        
        # Collect image details
        self.images = []
        self.labels = []
        
        for image_id in folder.get('image_list', []):
            image_doc = mongo_db.images.find_one({'_id': ObjectId(image_id)})
            if image_doc:
                # Retrieve image from GridFS
                image_file = gridfs_db.get(ObjectId(image_doc['image_id']))
                
                # Convert to PIL Image
                from PIL import Image
                import io
                image = Image.open(io.BytesIO(image_file.read()))
                
                self.images.append(image)
                # Assuming label is stored in image document, adjust as needed
                self.labels.append(image_doc.get('label', 0))
        
        self.classes = sorted(set(self.labels))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
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

# Model Definition (Similar to your previous implementation)
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
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
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_prob=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout1 = nn.Dropout(dropout_prob) if dropout_prob is not None else None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.dropout1 is not None:
            out = self.dropout1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, grayscale=False, dropout_prob=None):
        self.inplanes = 64
        in_dim = 1 if grayscale else 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout_prob=dropout_prob)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dropout_prob=dropout_prob)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dropout_prob=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout_prob=dropout_prob))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout_prob=dropout_prob))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

def ResNet18(num_classes, grayscale=False, dropout_prob=None):
    """Constructs a ResNet-18 model."""
    model = ResNet(block=BasicBlock, 
                   layers=[2, 2, 2, 2],
                   num_classes=num_classes,
                   grayscale=grayscale,
                   dropout_prob=dropout_prob
                   )
    return model

def train_model_with_multiple_folders(
    folder_ids: List[str], 
    mongo_db, 
    gridfs_db, 
    num_epochs: int = 10, 
    learning_rate: float = 0.001, 
    batch_size: int = 32,
    dropout_prob: Optional[float] = 0.2,
    val_split: float = 0.2,
    test_split: float = 0.1
) -> Dict[str, Any]:
    """
    Train a model on multiple folders with advanced configuration options.
    
    :param folder_ids: List of folder IDs to train on
    :param mongo_db: MongoDB database connection
    :param gridfs_db: GridFS connection
    :param num_epochs: Number of training epochs
    :param learning_rate: Learning rate for optimizer
    :param batch_size: Batch size for training
    :param dropout_prob: Dropout probability
    :param val_split: Validation split ratio
    :param test_split: Test split ratio
    :return: Dictionary containing training metrics
    """
    # Set random seed for reproducibility
    set_seed(42)

    # Prepare transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset
    try:
        full_dataset = MultifolderMongoGridFSDataset(
            folder_ids, 
            mongo_db, 
            gridfs_db, 
            transform=train_transform
        )
    except Exception as e:
        raise ValueError(f"Failed to create dataset: {e}")

    # Validate dataset
    if len(full_dataset) == 0:
        raise ValueError("No images found in the specified folders")

    # Get number of classes dynamically
    num_classes = len(full_dataset.classes)
    print(f"Found {num_classes} classes: {full_dataset.classes}")
    print(f"Total images: {len(full_dataset)}")

    # Split dataset
    test_size = int(test_split * len(full_dataset))
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - test_size - val_size

    test_dataset, val_dataset, train_dataset = random_split(full_dataset, [test_size, val_size, train_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model
    model = ResNet18(
        num_classes=num_classes, 
        grayscale=False, 
        dropout_prob=dropout_prob
    )

    # Prepare training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_losses, val_losses, best_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        num_epochs=num_epochs
    )

    # Evaluate on test set
    model.eval()
    test_correct = 0
    test_total = 0
    test_predictions = []
    test_true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits, probas = model(inputs)
            _, predicted = torch.max(probas, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            test_predictions.extend(predicted.cpu().numpy())
            test_true_labels.extend(labels.cpu().numpy())

    test_accuracy = test_correct / test_total

    # Prepare training metrics
    training_metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_accuracy': test_accuracy,
        'num_classes': num_classes,
        'class_names': full_dataset.classes,
        'class_distribution': {
            cls: sum(1 for label in full_dataset.labels if label == cls) 
            for cls in full_dataset.classes
        }
    }

    # Optionally save model and metrics
    model_save_path = os.path.join('models', f'model_{"_".join(folder_ids)}.pth')
    torch.save(best_model.state_dict(), model_save_path)

    return training_metrics

# Training Utilities
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    val_losses = []
    best_val_accuracy = 0.0
    best_model_weights = model.state_dict()
    
    start_time = time.time()
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        epoch_train_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, probas = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        average_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(average_train_loss)

        # Validation Phase
        model.eval()
        epoch_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                logits, probas = model(inputs)
                loss = criterion(logits, labels)
                epoch_val_loss += loss.item()

                _, predicted = torch.max(probas, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        average_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(average_val_loss)

        val_accuracy = correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {average_train_loss:.4f}, Val Loss: {average_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
        
        # Save the best model weights
        if val_accuracy > best_val_accuracy:
            print("New best model - Saving")
            best_val_accuracy = val_accuracy
            best_model_weights = copy.deepcopy(model.state_dict())

    # Load the best model weights
    model.load_state_dict(best_model_weights)

    print("Training complete. Best Validation Accuracy: {:.4f}".format(best_val_accuracy))

    return train_losses, val_losses, model

# Flask Routes for ML Training
@app.route('/train', methods=['POST'])
@jwt_required()
def train_ml_model():
    """
    API endpoint to train a machine learning model on a specific folder's images.
    
    Expected JSON payload:
    {
        'folder_id': 'mongodb_folder_id',
        'num_epochs': 10,
        'learning_rate': 0.001,
        'num_classes': 4,
        'dropout_prob': 0.2
    }
    """
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    folder_id = data.get('folder_id')
    num_epochs = data.get('num_epochs', 10)
    learning_rate = data.get('learning_rate', 1e-4)
    num_classes = data.get('num_classes', 4)
    dropout_prob = data.get('dropout_prob', 0.2)

    # Validate folder
    folder = mongo.db.folders.find_one({'_id': ObjectId(folder_id), 'user_id': current_user_id})
    if not folder:
        return jsonify({"error": "Folder not found or unauthorized"}), 404

    # Set random seed for reproducibility
    set_seed(42)

    # Prepare transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create dataset
    try:
        full_dataset = MongoGridFSDataset(
            folder_id, 
            mongo.db, 
            fs, 
            transform=train_transform
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Split dataset
    test_size = int(0.1 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - test_size - val_size

    test_dataset, val_dataset, train_dataset = random_split(full_dataset, [test_size, val_size, train_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Initialize model
    model = ResNet18(
        num_classes=num_classes, 
        grayscale=False, 
        dropout_prob=dropout_prob
    )

    # Prepare training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    try:
        train_losses, val_losses, best_model = train_model(
            model, 
            train_loader, 
            val_loader, 
            criterion, 
            optimizer, 
            num_epochs=num_epochs
        )

        # Save model to MongoDB (you might want to implement a more robust model saving mechanism)
        model_bytes = torch.save(best_model.state_dict(), r'D:\Pribadi\Sidehustle\p24_v2\backend_flask\models\best_model.pth')
        
        # Optional: Save training metrics
        training_metrics = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_accuracy': max(val_losses) 
            }

        # Save metrics to MongoDB (optional)
        mongo.db.models.insert_one({
            'folder_id': ObjectId(folder_id),
            'user_id': current_user_id,
            'training_metrics': training_metrics,
            'created_at': time.time()
        })

        return jsonify({"message": "Model trained successfully", "metrics": training_metrics}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/predict', methods=['POST']) 
@jwt_required() 
def predict(): 
    """ API endpoint to make predictions using a trained model.
        Expected JSON payload:
    {
        'model_id': 'mongodb_model_id',
        'image_id': 'mongodb_image_id'
    }
    """
    data = request.get_json()
    model_id = data.get('model_id')
    image_id = data.get('image_id')

    # Retrieve model
    model_doc = mongo.db.models.find_one({'_id': ObjectId(model_id)})
    if not model_doc:
        return jsonify({"error": "Model not found"}), 404

    # Retrieve number of classes from model_doc or fallback to 4
    num_classes = model_doc.get('num_classes', 4)
    dropout_prob = model_doc.get('dropout_prob', 0.2)

    # Load model
    model_path = r'D:\Pribadi\Sidehustle\p24_v2\backend_flask\models\best_model.pth'
    if not os.path.exists(model_path):
        return jsonify({"error": "Model file not found"}), 500

    # Initialize model with the same parameters used during training
    model = ResNet18(num_classes=num_classes, dropout_prob=dropout_prob)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Retrieve image
    try:
        image_file = fs.get(ObjectId(image_id))
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(image_file.read()))
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve image: {str(e)}"}), 500

    # Prepare image for model input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        logits, probas = model(image_tensor)
        _, predicted = torch.max(probas, 1)

    # Retrieve class mapping
    class_to_idx = model_doc.get('class_to_idx', {})
    
    # Map predicted class index to label
    predicted_class = list(class_to_idx.keys())[list(class_to_idx.values()).index(predicted.item())] \
        if predicted.item() in class_to_idx.values() else str(predicted.item())

    return jsonify({
        "predicted_class": predicted_class, 
        "probabilities": probas.tolist()
    }), 200

@app.route('/train_multi_folder', methods=['POST'])
@jwt_required()
def train_multi_folder_ml_model():
    """
    API endpoint to train a machine learning model on multiple folder IDs.
    
    Expected JSON payload:
    {
        'folder_ids': ['folder_id1', 'folder_id2', ...],
        'num_epochs': 10,
        'learning_rate': 0.001,
        'batch_size': 32,
        'dropout_prob': 0.2
    }
    """
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    folder_ids = data.get('folder_ids', [])
    num_epochs = data.get('num_epochs', 10)
    learning_rate = data.get('learning_rate', 1e-4)
    batch_size = data.get('batch_size', 32)
    dropout_prob = data.get('dropout_prob', 0.2)

    # Validate folders belong to the current user
    valid_folders = mongo.db.folders.find({
        '_id': {'$in': [ObjectId(fid) for fid in folder_ids]}, 
        'user_id': current_user_id
    })
    
    if valid_folders.count() != len(folder_ids):
        return jsonify({"error": "One or more folders not found or unauthorized"}), 404

    try:
        # Train the model
        training_metrics = train_model_with_multiple_folders(
            folder_ids, 
            mongo.db, 
            fs, 
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            dropout_prob=dropout_prob
        )

        # Optionally save metrics to database
        mongo.db.models.insert_one({
            'folder_ids': [ObjectId(fid) for fid in folder_ids],
            'user_id': current_user_id,
            'training_metrics': training_metrics,
            'created_at': time.time()
        })

        return jsonify({
            "message": "Model trained successfully", 
            "metrics": training_metrics
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Additional helper functions for model analysis can be added here
def analyze_model_performance(training_metrics):
    """
    Analyze and provide insights into model training performance.
    """
    insights = {
        'class_distribution': training_metrics['class_distribution'],
        'overall_performance': {
            'train_loss': training_metrics['train_losses'][-1],
            'val_loss': training_metrics['val_losses'][-1],
            'test_accuracy': training_metrics['test_accuracy']
        },
        'class_performance_hints': {}
    }

    # Basic class distribution analysis
    total_samples = sum(insights['class_distribution'].values())
    for cls, count in insights['class_distribution'].items():
        percentage = (count / total_samples) * 100
        if percentage < 10:
            insights['class_performance_hints'][cls] = "Low sample count - might need more data"
        elif percentage > 50:
            insights['class_performance_hints'][cls] = "Dominant class - potential class imbalance"

    return insights

@app.route('/predict_v2', methods=['POST'])
@jwt_required()
def predict_multi_folder_ml_model():
    """
    API endpoint to make predictions using a trained model.
    
    Expected JSON payload:
    {
        'model_id': 'specific_model_id',
        'image': 'base64_encoded_image'
    }
    """
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    # Validate input
    model_id = data.get('model_id')
    base64_image = data.get('image')
    
    if not model_id or not base64_image:
        return jsonify({"error": "Missing model_id or image"}), 400

    try:
        # Retrieve model metadata from database
        model_metadata = mongo.db.models.find_one({
            '_id': ObjectId(model_id),
            'user_id': current_user_id
        })
        
        if not model_metadata:
            return jsonify({"error": "Model not found or unauthorized"}), 404

        # Decode base64 image
        try:
            image_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            return jsonify({"error": f"Invalid image format: {str(e)}"}), 400

        # Prepare image transform (same as test transform in training)
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Prepare the image
        input_tensor = test_transform(image).unsqueeze(0)

        # Load the model
        model_path = os.path.join('models', f'model_{"_".join([str(fid) for fid in model_metadata["folder_ids"]])}.pth')
        num_classes = model_metadata['training_metrics']['num_classes']
        
        model = ResNet18(
            num_classes=num_classes, 
            grayscale=False, 
            dropout_prob=0.2  # Use the same dropout prob as during training
        )
        model.load_state_dict(torch.load(r'D:\Pribadi\Sidehustle\p24_v2\backend_flask\models\model_675a593707e1b45b9a0b5c0e_67595da1f8fa163aa08c3fa3.pth'))
        model.eval()

        # Make prediction
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            input_tensor = input_tensor.to(device)
            
            logits, probas = model(input_tensor)
            
            # Get top k predictions
            top_k = 5
            # Ensure we don't try to get more classes than exist
            top_k = min(top_k, num_classes)
            
            # Safety check for empty probabilities
            if probas.numel() == 0:
                return jsonify({"error": "Unable to generate predictions"}), 500

            # Safely get top k predictions
            top_k_prob, top_k_classes = torch.topk(probas, top_k)
            
            predictions = []
            class_names = model_metadata['training_metrics']['class_names']
            print(class_names)
            
            # Safely iterate through predictions
            for i in range(top_k_prob.size(1)):
                class_index = top_k_classes[0][i].item()
                print(class_index)
                # Ensure class index is valid
                if 0 <= class_index < len(class_names):
                    class_name = class_names[class_index]
                    probability = float(top_k_prob[0][i].item())
                    predictions.append({
                        'class': class_name,
                        'probability': probability
                    })

                print(predictions)

        # Ensure we have predictions
        if not predictions:
            return jsonify({"error": "No valid predictions could be generated"}), 500

        return jsonify({
            "predictions": predictions
        }), 200

    except Exception as e:
        # Log the full error for debugging
        app.logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
