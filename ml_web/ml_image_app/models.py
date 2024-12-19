# models.py
from django.db import models
from django.contrib.auth.models import User
import base64
from django.core.files.base import ContentFile
import uuid
from django.conf import settings
import requests

class ImageUpload(models.Model):    
    id = models.CharField(primary_key=True, max_length=36, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image_base64 = models.TextField()  # Store base64 encoded image
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file_name = models.CharField(max_length=255, blank=True, null=True)
    file_type = models.CharField(max_length=50, blank=True, null=True)

    def __str__(self):
        return f"Image {self.id} by {self.user.username}"

    def save_base64_image(self, image_file):
        """
        Convert uploaded image to base64 and save
        """
        # Read the image file
        if hasattr(image_file, 'read'):
            image_data = image_file.read()
        else:
            # If it's a path or file-like object
            with open(image_file, 'rb') as f:
                image_data = f.read()
        
        # Encode to base64
        self.image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Store additional metadata
        self.file_name = getattr(image_file, 'name', '')
        self.file_type = getattr(image_file, 'content_type', '')
        
        self.save()

    def get_image_file(self):
        """
        Decode base64 back to image file
        """
        if not self.image_base64:
            return None
        
        # Decode base64 to binary
        image_data = base64.b64decode(self.image_base64)
        
        # Create a file-like object
        return ContentFile(image_data, name=self.file_name)
    
class ImageService:
    """
    Service class to interact with the Flask image upload endpoint
    """
    BASE_URL = 'http://localhost:5000'  # Replace with your actual Flask backend URL

    @classmethod
    def upload_image(cls, request):
        """
        Upload an image to the Flask backend
        
        :param request: Django request object
        :return: Response from the backend
        """
        # Assuming you're using Django's authentication and have a way to get the JWT token
        jwt_token = request.session.get('access_token')  # Adjust based on how you store the token
        
        # Get the uploaded file from the request
        image_file = request.FILES.get('image')
        
        if not image_file:
            return None, "No image file found"
        
        # Prepare the files and headers for the request
        files = {'image': (image_file.name, image_file, image_file.content_type)}
        headers = {
            'Authorization': f'Bearer {jwt_token}'
        }
        
        try:
            # Make the POST request to the Flask backend
            response = requests.post(
                f'{cls.BASE_URL}/upload-image', 
                files=files, 
                headers=headers
            )
            
            # Check the response
            if response.status_code == 201:
                return response.json(), None
            else:
                return None, response.text
        
        except requests.RequestException as e:
            return None, str(e)

    @classmethod
    def get_images(cls, request):
        """
        Retrieve images for the current user
        
        :param request: Django request object
        :return: List of images or error
        """
        jwt_token = request.session.get('access_token')  # Adjust based on how you store the token
        
        headers = {
            'Authorization': f'Bearer {jwt_token}'
        }
        
        try:
            response = requests.get(
                f'{cls.BASE_URL}/images', 
                headers=headers
            )
            
            if response.status_code == 200:
                return response.json(), None
            else:
                return None, response.text
        
        except requests.RequestException as e:
            return None, str(e)
        
    @classmethod
    def get_image(cls, request, image_id):
        """
        Retrieve a specific image by ID
        
        :param request: Django request object
        :param image_id: ID of the image to retrieve
        :return: Image data or error
        """
        jwt_token = request.session.get('access_token')  # Adjust based on how you store the token
        
        headers = {
            'Authorization': f'Bearer {jwt_token}'
        }
        
        try:
            response = requests.get(
                f'{cls.BASE_URL}/image/{image_id}', 
                headers=headers
            )
            
            if response.status_code == 200:
                return response.content, None
            else:
                return None, response.text
        
        except requests.RequestException as e:
            return None, str(e)
        
    @classmethod
    def delete_image(cls, request, image_id):
        """
        Delete an image by ID
        
        :param request: Django request object
        :param image_id: ID of the image to delete
        :return: Success message or error
        """
        jwt_token = request.session.get('access_token')  # Adjust based on how you store the token

        headers = {
            'Authorization': f'Bearer {jwt_token}'
        }

        try:
            response = requests.delete(
                f'{cls.BASE_URL}/image/{image_id}',
                headers=headers
            )

            if response.status_code == 200:
                return response.json(), None
            else:
                return None, response.text
            
        except requests.RequestException as e:
            return None, str(e)
        

    @classmethod
    def create_folder(cls, request, folder_name):
        """
        Create a new folder
        
        :param request: Django request object
        :param folder_name: Name of the folder to create
        :return: Created folder details or error
        """
        jwt_token = request.session.get('access_token')  # Adjust based on how you store the token
        
        headers = {
            'Authorization': f'Bearer {jwt_token}',
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(
                f'{cls.BASE_URL}/create-folder', 
                json={'folder_name': folder_name},
                headers=headers
            )
            
            if response.status_code == 201:
                return response.json(), None
            else:
                return None, response.text
        
        except requests.RequestException as e:
            return None, str(e)