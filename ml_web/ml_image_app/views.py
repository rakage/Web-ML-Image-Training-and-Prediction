# views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.http import JsonResponse, HttpResponseForbidden
from .forms import ImageUploadForm, BulkImageUploadForm
from .models import ImageUpload, ImageService
from .ml_utils import detect_objects, train_model  # Machine learning functions
from .tasks import process_image, retrain_model  # Celery tasks for async processing
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
import logging
from django.contrib import messages
import requests
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.csrf import ensure_csrf_cookie
from django.conf import settings
import json
from PIL import Image
import base64
import io
import uuid
from ml_web.custom_auth import FlaskAuthBackend
from django.core.files.uploadedfile import InMemoryUploadedFile
import mimetypes

logger = logging.getLogger(__name__)
FLASK_API_URL = 'http://localhost:5000'
# Homepage view
@login_required
def homepage(request):
    # if 'access_token' not in request.session:
    #     return redirect('login')
    
    # images = ImageUpload.objects.filter(user=request.user)
    return render(request, 'home.html')


# def register(request):
#     if request.method == 'POST':
#         form = UserCreationForm(request.POST)
#         if form.is_valid():
#             user = form.save()
#             login(request, user)
#             return redirect('home')
#         else:
#             # Log form errors for debugging
#             logger.error(f"Registration form errors: {form.errors}")
#             print(f"Registration form errors: {form.errors}")
#     else:
#         form = UserCreationForm()
    
#     # Log template rendering
#     logger.info("Rendering register template")
#     print("Rendering register template")
#     return render(request, 'register.html', {'form': form})

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            try:
                # Send registration request to Flask backend
                response = requests.post(f'{settings.FLASK_AUTH_BACKEND_URL}/register', json={
                    'username': form.cleaned_data['username'],
                    'password': form.cleaned_data['password1']
                })
                
                if response.status_code == 201:  # Successful registration
                    # Authenticate and login user using the custom backend
                    user = authenticate(
                        request, 
                        username=form.cleaned_data['username'], 
                        password=form.cleaned_data['password1']
                    )
                    
                    if user is not None:
                        login(request, user)
                        return redirect('home')
                    else:
                        # Log authentication failure
                        logger.error("Authentication failed after registration")
                        form.add_error(None, "Registration successful but login failed")
                else:
                    # Handle registration error from Flask backend
                    error_msg = response.json().get('message', 'Registration failed')
                    form.add_error(None, error_msg)
            
            except requests.RequestException as e:
                logger.error(f"Registration request error: {e}")
                form.add_error(None, "Network error occurred during registration")
        
        # Log form errors for debugging
        if form.errors:
            logger.error(f"Registration form errors: {form.errors}")
            print(f"Registration form errors: {form.errors}")
    else:
        form = UserCreationForm()
    
    # Log template rendering
    logger.info("Rendering register template")
    print("Rendering register template")
    return render(request, 'register.html', {'form': form})

# @csrf_exempt  # Temporarily disable CSRF for this endpoint (use with caution in production)
# def save_token(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)  # Parse JSON data from the request body
#             access_token = data.get('access_token')

#             if access_token:
#                 request.session['access_token'] = access_token
#                 return JsonResponse({'status': 'success'})
#             else:
#                 return JsonResponse({'status': 'failure', 'message': 'No access token provided'}, status=400)
#         except Exception as e:
#             return JsonResponse({'status': 'failure', 'message': str(e)}, status=500)
#     return JsonResponse({'status': 'failure', 'message': 'Invalid request method'}, status=405)

@csrf_exempt  # Use with caution in production
def save_token(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            access_token = data.get('access_token')

            if access_token:
                # Store token in session if needed
                request.session['access_token'] = access_token
                return JsonResponse({'status': 'success'})
            else:
                return JsonResponse({'status': 'failure', 'message': 'No access token provided'}, status=400)
        except Exception as e:
            return JsonResponse({'status': 'failure', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'failure', 'message': 'Invalid request method'}, status=405)



# Login view
# def login_view(request):
#     if 'access_token' in request.session:
#         return redirect('')
    
#     if request.method == 'POST':
#         form = AuthenticationForm(request, data=request.POST)
#         if form.is_valid():
#             print('masuk 1')
#             # Login the user via Flask API
#             data = {
#                 'username': form.cleaned_data['username'],
#                 'password': form.cleaned_data['password'],
#             }
#             response = requests.post(f'{FLASK_API_URL}/login', json=data)
#             print(response.json())
            
#             if response.status_code == 200:  # Login successful
#                 # Assuming the API returns a JWT token
#                 access_token = response.json().get('access_token')
#                 # Set the token in Django session or cookies
#                 request.session['access_token'] = access_token
#                 return redirect('')
#             else:
#                 # Handle error (invalid credentials)
#                 error_message = response.json().get('msg', 'Error occurred')
#                 return render(request, 'login.html', {'form': form, 'error_message': error_message})
#     else:
#         form = AuthenticationForm()
    
#     return render(request, 'login.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            
            # Use Django's authenticate method which now uses our custom backend
            user = authenticate(request, username=username, password=password)
            
            if user is not None:
                # User authentication successful
                login(request, user)
                return redirect('home')
            else:
                # Authentication failed
                form.add_error(None, 'Invalid username or password')
    else:
        form = AuthenticationForm()
    
    return render(request, 'login.html', {'form': form})

# Logout view
# def logout_view(request):
#     logout(request)
#     return redirect('login')

def logout_view(request):
    # Optional: Add logout request to Flask backend if needed
    try:
        # You might want to send a logout request to your Flask backend
        requests.post(f'{FLASK_API_URL}/logout', 
                      headers={'Authorization': f'Bearer {request.session.get("access_token")}'})
    except requests.RequestException:
        # Log the error but don't block logout
        logger.warning("Failed to logout from backend")
    
    # Django logout
    logout(request)
    return redirect('login')

@login_required
def upload_image(request):
    auth_backend = FlaskAuthBackend()
    jwt_token = auth_backend.get_access_token(request.user)
    if request.method == 'POST':
        # Check if it's a standard file upload
        form = BulkImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Process multiple image uploads
            uploaded_images = request.FILES.getlist('images')
            folder_id = request.POST.get('folder_id')
            successful_uploads = 0
            failed_uploads = []

            for image_file in uploaded_images:
                try:
                    result = _process_image_upload(request, image_file, folder_id)
                    # If the result is successful (assuming it returns True or a success response)
                    successful_uploads += 1
                except Exception as e:
                    failed_uploads.append((image_file.name, str(e)))
            
            # Provide feedback about uploads
            if successful_uploads > 0:
                messages.success(request, f'{successful_uploads} image(s) uploaded successfully!')
                return JsonResponse({'status': 'success', 'message': 'Image uploaded successfully!'})
            
            if failed_uploads:
                error_message = "Failed to upload the following images:\n"
                for filename, error in failed_uploads:
                    error_message += f"- {filename}: {error}\n"
                messages.error(request, error_message)
                return JsonResponse({'status': 'error', 'message': error_message})

            
        
        elif 'webcam_image' in request.POST:
            # Webcam image capture
            webcam_image_data = request.POST['webcam_image']
            
            # Remove the data URL prefix
            if webcam_image_data.startswith('data:image/jpeg;base64,'):
                webcam_image_data = webcam_image_data.split(',')[1]
            
            try:
                # Decode base64 image
                image_data = base64.b64decode(webcam_image_data)
                
                # Generate a unique filename
                filename = f'webcam_capture_{uuid.uuid4().hex}.jpg'
                
                # Create a file-like object
                image_file = io.BytesIO(image_data)
                
                # Use Django's InMemoryUploadedFile to create a file-like object that mimics request.FILES
                from django.core.files.uploadedfile import InMemoryUploadedFile
                image_file = InMemoryUploadedFile(
                    file=image_file,
                    field_name='image',
                    name=filename,
                    content_type='image/jpeg',
                    size=len(image_data),
                    charset=None
                )
                
                result = _process_image_upload(request, image_file, folder_id)
                return JsonResponse({'status': 'success', 'message': 'Image uploaded successfully!'})

            except Exception as e:
                return JsonResponse({'status': 'error', 'message': f'Error processing webcam image: {str(e)}'})
        
        else:
            # No image found in the request
            messages.error(request, 'No image uploaded.')
            form = ImageUploadForm()
    
    else:
        form = ImageUploadForm()

    return render(request, 'upload.html', {'form': form,
                                            'auth_token': jwt_token
                                            })
                                            

def _process_image_upload(request, image_file, folder_id):
    """
    Common method to process image upload for both file upload and webcam capture
    """
    # Getting the JWT token for the logged-in user (assuming your FlaskAuthBackend handles token fetching)
    auth_backend = FlaskAuthBackend()
    jwt_token = auth_backend.get_access_token(request.user)

    # Prepare the request to Flask API
    headers = {
        'Authorization': f'Bearer {jwt_token}'  # Include the token in the header
    }

    try:
        # Prepare files for upload
        # Use the original filename or generate a unique one
        filename = image_file.name if hasattr(image_file, 'name') else f'upload_{uuid.uuid4().hex}.jpg'
        
        # Determine the content type
        content_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        
        # Prepare the files dictionary for requests
        files = {'image': (filename, image_file, content_type)}
        # folder_id = request.POST.get('folder_id')
        print(folder_id)

        # Make the POST request to Flask backend
        response = requests.post(f'{FLASK_API_URL}/upload-image', files=files, headers=headers, data={'folder_id': folder_id})
        print(response)

        if response.status_code == 201:
            # If the image is successfully uploaded, redirect to home
            # messages.success(request, 'Image uploaded successfully!')
            return redirect('upload')
        else:
            # If the upload failed, show an error message
            error_msg = response.json().get('msg', 'Failed to upload image.')
            messages.error(request, f'Upload failed: {error_msg}')
    
    except requests.RequestException as e:
        # Catch any exception during the request
        messages.error(request, f'Network error: {str(e)}')
    except Exception as e:
        # Catch any other unexpected errors
        messages.error(request, f'Unexpected error: {str(e)}')
    
    return redirect('upload')


@login_required
def list_images_view(request):
    """
    View to list user's images
    """
    
    auth_backend = FlaskAuthBackend()
    jwt_token = auth_backend.get_access_token(request.user)

    headers = {
        'Authorization': f'Bearer {jwt_token}'
    }

    try:
        response = requests.get(f'{FLASK_API_URL}/images', headers=headers)
        
        if response.status_code == 200:
            images = response.json()
            return render(request, 'image_list.html', {
                'images': images,
                'auth_token': jwt_token
            })
        else:
            return render(request, 'error.html', {
                'error': response.text
            })

    except requests.RequestException as e:
        return render(request, 'error.html', {
            'error': str(e)
        })
    

@login_required
def folders_list_view(request):
    """
    View to list user's folders
    """
    
    auth_backend = FlaskAuthBackend()
    jwt_token = auth_backend.get_access_token(request.user)

    headers = {
        'Authorization': f'Bearer {jwt_token}'
    }

    try:
        response = requests.get(f'{FLASK_API_URL}/folders', headers=headers)
        
        if response.status_code == 200:
            folders = response.json()
            return render(request, 'folders_list.html', {
                'folders': folders,
                'auth_token': jwt_token
            })
        else:
            return render(request, 'error.html', {
                'error': response.text
            })

    except requests.RequestException as e:
        return render(request, 'error.html', {
            'error': str(e)
        })
    
@login_required
def folder_detail_view(request, folder_id):
    """
    View to show details of a specific folder
    """
    
    auth_backend = FlaskAuthBackend()
    jwt_token = auth_backend.get_access_token(request.user)

    headers = {
        'Authorization': f'Bearer {jwt_token}'
    }

    try:
        response = requests.get(f'{FLASK_API_URL}/folder/{folder_id}', headers=headers)
        
        if response.status_code == 200:
            folder = response.json()
            return render(request, 'folder_detail.html', {
                'folder': folder,
                'auth_token': jwt_token
            })
        else:
            return render(request, 'error.html', {
                'error': response.text
            })

    except requests.RequestException as e:
        return render(request, 'error.html', {
            'error': str(e)
        })

# Model training view
@login_required
def train_model_view(request):
    if request.method == 'POST':
        retrain_model.apply_async()
        return redirect('home')
    return render(request, 'model_training.html')

@login_required
def predict_view(request):
    auth_backend = FlaskAuthBackend()
    jwt_token = auth_backend.get_access_token(request.user)

    endpoint_model_options = 'http://127.0.0.1:5000//list_models'
    headers = {
        'Authorization': f'Bearer {jwt_token}'
    }

    try:
        response = requests.get(endpoint_model_options, headers=headers)
        
        if response.status_code == 200:
            model_options = response.json()
        else:
            return render(request, 'error.html', {
                'error': response.text
            })
        
    except requests.RequestException as e:
        return render(request, 'error.html', {
            'error': str(e)
        })
        

    context = {
        "model_options": model_options,
        "predictions": None,
        "predicted_image": None,
        "error": None
    }

    if request.method == "POST":
        # Get the selected model
        selected_model = request.POST.get("model")
        image_file = request.FILES.get("image")

        if not selected_model or not image_file:
            context["error"] = "Please provide an image and select a model."
            return render(request, "predict.html", context)

        # Send the image and model to the prediction endpoint
        try:
            endpoint_url = "http://127.0.0.1:5000/predict"  # Replace with your endpoint URL
            response = requests.post(
                endpoint_url,
                headers=headers,
                files={"image": image_file},
                data={"model": selected_model},
            )

            if response.status_code == 200:
                response_data = response.json()
                context["predictions"] = response_data.get("predictions", [])
                image_base64 = response_data.get("image")
                if image_base64:
                    context["predicted_image"] = f"data:image/jpeg;base64,{image_base64}"
            else:
                context["error"] = f"Error: {response.status_code} - {response.text}"

        except Exception as e:
            context["error"] = f"Error connecting to the prediction endpoint: {e}"

    return render(request, "predict.html", context)


def save_bboxes_view(request):
    auth_backend = FlaskAuthBackend()
    jwt_token = auth_backend.get_access_token(request.user)
    headers = {
        'Authorization': f'Bearer {jwt_token}'
    }

    if request.method == 'POST':
        data = json.loads(request.body)
        image_id = data.get('image_id')
        bboxes = data.get('bboxes')

        if not image_id or not bboxes:
            return JsonResponse({'status': 'error', 'message': 'Invalid data provided'}, status=400)
        
        # Save the bounding boxes to the database
        try:
            payload = {
                'image_id': image_id,
                'bboxes': bboxes
            }

            response = requests.post(f'{FLASK_API_URL}/save-bboxes', json=payload, headers=headers) 
            if response.status_code == 200:
                return JsonResponse({'status': 'success'})
            else:
                return JsonResponse({'status': 'error', 'message': response.text}, status=response.status_code)
            
        except requests.RequestException as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
        
    else:
        try:
            response = requests.get(f'{FLASK_API_URL}/folders', headers=headers)
            
            if response.status_code == 200:
                folders = response.json()
                return render(request, 'save_bbox.html', {
                'folder': folders,
                'auth_token': jwt_token
                })
            else:
                return render(request, 'error.html', {
                    'error': response.text
                })
            
        except requests.RequestException as e:
            return render(request, 'error.html', {
                'error': str(e)
            })

