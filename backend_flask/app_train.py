from flask import Flask, jsonify, Response, request
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity
from pymongo import MongoClient
import gridfs
from bson import ObjectId
import threading
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from flask_pymongo import PyMongo
from datetime import datetime

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'caowfjca9812321kfafqw'  # Replace with your JWT secret
app.config['MONGO_URI'] = 'mongodb://localhost:27017/mlweb_v2'
jwt = JWTManager(app)

# Initialize MongoDB client
mongo = PyMongo(app)
fs = gridfs.GridFS(mongo.db)

# Global dictionary to store training progress
training_progress = {}

# Define the Folder-Based Dataset
class FolderBasedDataset(Dataset):
    def __init__(self, db, fs, user_id, folder_ids, transform=None):
        self.db = db
        self.fs = fs
        self.user_id = ObjectId(user_id)
        self.transform = transform
        self.folder_ids = folder_ids
        
        # Create a mapping from folder_id to label
        self.label_map = {str(folder_id): idx for idx, folder_id in enumerate(folder_ids)}
        
        # Fetch all image records from the specified folders
        self.image_records = []
        for folder_id in folder_ids:
            folder = self.db.folders.find_one({'_id': ObjectId(folder_id)})
            if folder:
                for image_id in folder.get('image_list', []):
                    image_record = self.db.images.find_one({'_id': ObjectId(image_id)})
                    if image_record:
                        self.image_records.append({
                            'image_id': str(image_record['image_id']),
                            'folder_id': str(folder_id)
                        })
        print(f"Total image records found: {len(self.image_records)}")

        for record in self.image_records[:5]:  # Check first 5 records
            try:
                file_id = record['image_id']
                file = self.fs.get(ObjectId(file_id))
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                print(f"Successfully loaded image: {file_id}")
            except Exception as e:
                print(f"Error loading image {file_id}: {e}")

    def __len__(self):
        return len(self.image_records)
    
    def __getitem__(self, idx):
        image_record = self.image_records[idx]
        file_id = image_record['image_id']
        folder_id = image_record['folder_id']
        label = self.label_map[folder_id]
        
        # Fetch the image file from GridFS
        file = self.fs.get(ObjectId(file_id))
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_model(num_classes):
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(64 * 56 * 56, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model

def save_model_to_gridfs(model, user_id, fs, db):
    """
    Save model weights to GridFS and store metadata in the database
    
    Args:
    - model: Trained PyTorch model
    - user_id: ID of the user
    - fs: GridFS instance
    - db: MongoDB database instance
    """
    # Determine the number of classes from the last layer
    num_classes = model[-1].out_features  # Assuming the last layer is the classification layer
    
    # Serialize model weights to a BytesIO object
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    
    # Save to GridFS
    file_id = fs.put(
        buffer, 
        filename=f'model_{user_id}.pth',
        content_type='application/octet-stream',
        metadata={
            'user_id': user_id,
            'model_type': 'image_classification'
        }
    )
    
    # Store reference in a models collection with the number of classes
    db.models.insert_one({
        'user_id': user_id,
        'file_id': file_id,
        'created_at': datetime.utcnow(),
        'num_classes': num_classes
    })
    
    return file_id

def load_model_from_gridfs(user_id, fs, db):
    """
    Load model weights from GridFS for a specific user
    
    Args:
    - user_id: ID of the user
    - fs: GridFS instance
    - db: MongoDB database instance
    
    Returns:
    - Loaded PyTorch model
    """
    # Find the most recent model for the user
    model_record = db.models.find_one(
        {'user_id': user_id}, 
        sort=[('created_at', -1)]
    )
    
    if not model_record:
        raise ValueError("No model found for the user")
    
    # Retrieve the file from GridFS
    file = fs.get(model_record['file_id'])
    
    # Load the model weights
    buffer = io.BytesIO(file.read())
    num_classes = model_record.get('num_classes', 2)  # Default to 2 if not specified
    
    # Recreate the model with the EXACT number of classes used during training
    model = create_model(num_classes)
    model.load_state_dict(torch.load(buffer))
    model.eval()  # Set to evaluation mode
    
    return model

@app.route('/train', methods=['POST'])
@jwt_required()
def start_training():
    user_id = get_jwt_identity()
    
    data = request.get_json()
    if not data or 'folder_ids' not in data:
        return jsonify({'msg': 'Missing folder_ids in request body.'}), 400
    
    folder_ids = data['folder_ids']
    if not isinstance(folder_ids, list) or not folder_ids:
        return jsonify({'msg': 'folder_ids must be a non-empty list.'}), 400
    
    # Validate folder_ids: ensure they exist and belong to the user
    valid_folder_ids = []
    for fid in folder_ids:
        if ObjectId.is_valid(fid):
            folder = mongo.db.folders.find_one({'_id': ObjectId(fid)})
            if folder:
                valid_folder_ids.append(ObjectId(fid))
    
    if not valid_folder_ids:
        return jsonify({'msg': 'No valid folders found for the user.'}), 404
    
    # Create the model with the correct number of classes
    model = create_model(len(valid_folder_ids))
    
    # Train the model (your existing train_model function)
    train_model(user_id, valid_folder_ids)
    
    # Save to GridFS with the correct number of classes
    file_id = save_model_to_gridfs(model, user_id, fs, mongo.db)
    
    return jsonify({
        'msg': 'Training completed and model saved.', 
        'file_id': str(file_id)
    }), 200

# Training function
# def train_model(user_id, folder_ids):
#     """
#     Enhanced training function with more detailed error logging and debugging.
#     """
#     # Ensure the training progress is reset
#     training_progress[str(user_id)] = {
#         'status': 'running',
#         'epochs': [],
#         'final_result': None,
#         'debug_info': []  # Added debug info tracking
#     }
    
#     try:
#         # More verbose logging of input parameters
#         print(f"Training for user_id: {user_id}")
#         print(f"Folder IDs: {folder_ids}")
        
#         # Define data transformations
#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#         ])
        
#         # Initialize the dataset and dataloader
#         dataset = FolderBasedDataset(mongo.db, fs, user_id, folder_ids, transform=transform)
        
#         # Debug: Check dataset details
#         print(f"Dataset length: {len(dataset)}")
        
#         # If dataset is empty, log and return
#         if len(dataset) == 0:
#             error_msg = "No images found in the specified folders."
#             training_progress[str(user_id)]['status'] = 'failed'
#             training_progress[str(user_id)]['final_result'] = error_msg
#             training_progress[str(user_id)]['debug_info'].append(error_msg)
#             return
        
#         # Detailed logging of image records
#         for idx, record in enumerate(dataset.image_records[:5]):  # Log first 5 records
#             print(f"Record {idx}: {record}")
        
#         dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
#         # Debug: Check first batch
#         try:
#             first_batch = next(iter(dataloader))
#             print(f"First batch shape: {first_batch[0].shape}")
#             print(f"First batch labels: {first_batch[1]}")
#         except Exception as batch_error:
#             print(f"Error processing first batch: {batch_error}")
#             training_progress[str(user_id)]['debug_info'].append(f"Batch loading error: {batch_error}")
        
#         # Determine the number of classes
#         num_classes = len(folder_ids)
#         print(f"Number of classes: {num_classes}")
        
#         # Define a simple neural network
#         model = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(224*224*3, 512),
#             nn.ReLU(),
#             nn.Linear(512, num_classes)
#         )
        
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=0.001)
        
#         epochs = 10
#         for epoch in range(1, epochs + 1):
#             running_loss = 0.0
#             epoch_data = {
#                 'number': epoch,
#                 'batches': []
#             }
            
#             for batch_idx, (inputs, labels) in enumerate(dataloader, 1):
#                 # Add more detailed error handling
#                 try:
#                     optimizer.zero_grad()
#                     outputs = model(inputs)
#                     loss = criterion(outputs, labels)
#                     loss.backward()
#                     optimizer.step()
                    
#                     running_loss += loss.item()
                    
#                     if batch_idx % 10 == 0:
#                         avg_loss = running_loss / 10
#                         running_loss = 0.0
                        
#                         batch_info = {
#                             'batch': batch_idx,
#                             'loss': avg_loss
#                         }
#                         epoch_data['batches'].append(batch_info)
#                 except Exception as batch_error:
#                     print(f"Error in batch {batch_idx} of epoch {epoch}: {batch_error}")
#                     training_progress[str(user_id)]['debug_info'].append(
#                         f"Batch {batch_idx} error: {batch_error}"
#                     )
            
#             training_progress[str(user_id)]['epochs'].append(epoch_data)

#         # Training complete
#         training_progress[str(user_id)]['status'] = 'completed'
#         training_progress[str(user_id)]['final_result'] = 'Training completed successfully'
    
#     except Exception as e:
#         # Comprehensive error logging
#         error_msg = f"Training failed: {str(e)}"
#         print(error_msg)
#         import traceback
#         traceback.print_exc()  # Print full stack trace
        
#         training_progress[str(user_id)]['status'] = 'failed'
#         training_progress[str(user_id)]['final_result'] = error_msg

def train_model(user_id, folder_ids):
    """
    Enhanced training function with detailed batch logging and progress tracking.
    """
    # Ensure the training progress is reset
    training_progress[str(user_id)] = {
        'status': 'running',
        'epochs': [],
        'final_result': None,
        'debug_info': []
    }
    
    try:
        # More verbose logging of input parameters
        print(f"Training for user_id: {user_id}")
        print(f"Folder IDs: {folder_ids}")
        
        # Define data transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize the dataset and dataloader
        dataset = FolderBasedDataset(mongo.db, fs, user_id, folder_ids, transform=transform)
        
        # Debug: Check dataset details
        print(f"Dataset length: {len(dataset)}")
        
        # If dataset is empty, log and return
        if len(dataset) == 0:
            error_msg = "No images found in the specified folders."
            training_progress[str(user_id)]['status'] = 'failed'
            training_progress[str(user_id)]['final_result'] = error_msg
            training_progress[str(user_id)]['debug_info'].append(error_msg)
            return
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Determine the number of classes
        num_classes = len(folder_ids)
        print(f"Number of classes: {num_classes}")
        
        # Define a more complex neural network
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        print(f"Learning rate scheduler: {scheduler}")
        
        epochs = 10
        for epoch in range(1, epochs + 1):
            model.train()  # Set model to training mode
            running_loss = 0.0
            correct = 0
            total = 0
            
            epoch_data = {
                'number': epoch,
                'batches': [],
                'total_loss': 0.0,
                'accuracy': 0.0
            }
            
            for batch_idx, (inputs, labels) in enumerate(dataloader, 1):
                try:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    # Compute running loss and accuracy
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Log batch information every 10 batches
                    if batch_idx % 1 == 0:
                        avg_loss = running_loss / max(1, batch_idx)
                        batch_accuracy = 100 * correct / total if total > 0 else 0
                        print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {avg_loss}, Accuracy: {batch_accuracy}")
                        
                        batch_info = {
                            'batch': batch_idx,
                            'loss': avg_loss,
                            'accuracy': batch_accuracy
                        }
                        epoch_data['batches'].append(batch_info)
                        
                        # Reset running metrics
                        running_loss = 0.0
                        correct = 0
                        total = 0
                
                except Exception as batch_error:
                    print(f"Error in batch {batch_idx} of epoch {epoch}: {batch_error}")
                    training_progress[str(user_id)]['debug_info'].append(
                        f"Batch {batch_idx} error: {batch_error}"
                    )
            
            # End of epoch processing
            scheduler.step()
            
            # Compute epoch-level metrics
            epoch_data['total_loss'] = sum(batch['loss'] for batch in epoch_data['batches'])
            epoch_data['accuracy'] = sum(batch['accuracy'] for batch in epoch_data['batches']) / len(epoch_data['batches']) if epoch_data['batches'] else 0
            
            training_progress[str(user_id)]['epochs'].append(epoch_data)

        # Training complete
        training_progress[str(user_id)]['status'] = 'completed'
        training_progress[str(user_id)]['final_result'] = 'Training completed successfully'
        
        # Optional: Save the model
        torch.save(model.state_dict(), f'model_{user_id}.pth')
    
    except Exception as e:
        # Comprehensive error logging
        error_msg = f"Training failed: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()  # Print full stack trace
        
        training_progress[str(user_id)]['status'] = 'failed'
        training_progress[str(user_id)]['final_result'] = error_msg

# Endpoint to start training
# @app.route('/train', methods=['POST'])
# @jwt_required()
# def start_training():
#     """
#     Starts the training process for the authenticated user based on specified folders.
#     Expects a JSON payload with 'folder_ids': [<folder_id1>, <folder_id2>, ...]
#     """
#     user_id = get_jwt_identity()
    
#     data = request.get_json()
#     if not data or 'folder_ids' not in data:
#         return jsonify({'msg': 'Missing folder_ids in request body.'}), 400
    
#     folder_ids = data['folder_ids']
#     if not isinstance(folder_ids, list) or not folder_ids:
#         return jsonify({'msg': 'folder_ids must be a non-empty list.'}), 400
    
#     # Validate folder_ids: ensure they exist and belong to the user
#     valid_folder_ids = []
#     for fid in folder_ids:
#         if ObjectId.is_valid(fid):
#             folder = mongo.db.folders.find_one({'_id': ObjectId(fid)})
#             if folder:
#                 valid_folder_ids.append(ObjectId(fid))
    
#     if not valid_folder_ids:
#         return jsonify({'msg': 'No valid folders found for the user.'}), 404
    
#     train_models = train_model(user_id, valid_folder_ids)
    
#     return jsonify({'msg': 'Training started.'}), 202

# Endpoint to check training progress
@app.route('/train/progress', methods=['GET'])
@jwt_required()
def get_training_progress():
    """
    Retrieve the current training progress for the authenticated user.
    """
    user_id = get_jwt_identity()
    
    progress = training_progress.get(str(user_id), {
        'status': 'not_started',
        'epochs': [],
        'final_result': None
    })
    
    return jsonify(progress), 200

@app.route('/predict', methods=['POST'])
@jwt_required()
def predict_image():
    """
    Predict the class of an uploaded image using the user's trained model
    """
    user_id = get_jwt_identity()

    # Check if image is in the request
    if 'image' not in request.files:
        return jsonify({'msg': 'No image uploaded'}), 400
    
    image_file = request.files['image']
    
    try:
        # Load the user's model
        model = load_model_from_gridfs(user_id, fs, mongo.db)
        
        # Prepare image transformation (same as during training)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Open and transform the image
        image = Image.open(image_file).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Retrieve folder information to map predictions to labels
        model_record = mongo.db.models.find_one(
            {'user_id': user_id}, 
            sort=[('created_at', -1)]
        )
        
        if not model_record:
            return jsonify({'msg': 'No model found for the user'}), 404
        
        # Get the folders used for training to map predictions
        training_folders = mongo.db.folders.find({
            '_id': {'$in': model_record.get('folder_ids', [])}
        })
        
        # Create label mapping
        label_map = {idx: str(folder['_id']) for idx, folder in enumerate(training_folders)}
        
        # Perform prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class_id = predicted.item()
        
        # Get the corresponding folder ID
        predicted_folder_id = label_map.get(predicted_class_id, 'Unknown')
        
        return jsonify({
            'predicted_folder_id': predicted_folder_id,
            'confidence': float(torch.max(torch.softmax(outputs, dim=1)).item())
        }), 200
    
    except Exception as e:
        return jsonify({'msg': f'Prediction error: {str(e)}'}), 500

# Existing image retrieval endpoints
@app.route('/image/<image_id>', methods=['GET'])
@jwt_required()
def get_image(image_id):
    current_user_id = get_jwt_identity()

    # Validate ObjectId
    if not ObjectId.is_valid(image_id):
        return jsonify({"msg": "Invalid image ID."}), 400

    # Find the image in GridFS
    image_data = mongo.db.images.find_one({'_id': ObjectId(image_id), 'user_id': ObjectId(current_user_id)})
    
    if not image_data:
        return jsonify({"msg": "Image not found"}), 404

    # Fetch the image from GridFS
    file = fs.get(ObjectId(image_data['image_id']))
    
    # Determine the content type based on file metadata or extension
    content_type = image_data.get('content_type', 'image/png')  # Default to PNG
    
    # Return the image file as a response
    return Response(file.read(), content_type=content_type)

@app.route('/folder/<folder_id>', methods=['GET'])
@jwt_required()
def get_folder_images(folder_id):
    current_user_id = get_jwt_identity()
    
    # Validate ObjectId
    if not ObjectId.is_valid(folder_id):
        return jsonify({"msg": "Invalid folder ID."}), 400
    
    # Find the folder
    folder = mongo.db.folders.find_one({'_id': ObjectId(folder_id), 'user_id': current_user_id})
    if not folder:
        return jsonify({"msg": "Folder not found"}), 404
    
    image_details = []
    for image_id in folder.get('image_list', []):
        if ObjectId.is_valid(str(image_id)):
            image = mongo.db.images.find_one({'_id': ObjectId(image_id), 'user_id': current_user_id})
            if image:
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

# Endpoint to list all folders for the user
@app.route('/folders', methods=['GET'])
@jwt_required()
def list_folders():
    current_user_id = get_jwt_identity()
    
    folders_cursor = mongo.db.folders.find({'user_id': ObjectId(current_user_id)})
    folders = []
    for folder in folders_cursor:
        folders.append({
            'folder_id': str(folder['_id']),
            'folder_name': folder['folder_name'],
            'created_at': folder['created_at'],
            'image_count': len(folder.get('image_list', []))
        })
    
    return jsonify({'folders': folders}), 200

# Run the Flask app
if __name__ == '__main__':
    app.run()