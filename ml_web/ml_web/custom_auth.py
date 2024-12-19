import requests
from django.contrib.auth.backends import BaseBackend
from django.contrib.auth import get_user_model
from django.conf import settings
from django.contrib.auth.models import update_last_login
from django.core.cache import cache

User = get_user_model()

class FlaskAuthBackend(BaseBackend):
    """
    Custom authentication backend that authenticates against a Flask backend
    and manages JWT tokens
    """
    def authenticate(self, request, username=None, password=None):
        """
        Authenticate against the Flask backend and manage JWT token
        
        :param request: HttpRequest object
        :param username: Username to authenticate
        :param password: Password to authenticate
        :return: User object if authentication is successful, None otherwise
        """
        try:
            # Replace with your Flask backend login endpoint
            response = requests.post(
                f'{settings.FLASK_AUTH_BACKEND_URL}/login',
                json={
                    'username': username,
                    'password': password
                }
            )
            print('response',response)
            # Check if authentication was successful
            if response.status_code == 200:
                # Get authentication response data
                auth_data = response.json()
                
                # Extract JWT tokens
                access_token = auth_data.get('access_token')
                
                
                if not access_token:
                    return None
                
                # Try to get existing user or create a new one
                try:
                    user = User.objects.get(username=username)
                except User.DoesNotExist:
                    # Create a new user if not exists
                    user = User.objects.create_user(
                        username=username,
                        password=password  # This ensures Django can later validate the password
                    )
                
                # Save tokens securely
                self.save_tokens(user, access_token)
                
                # Update last login
                update_last_login(None, user)
                
                return user
            
            return None
        
        except requests.RequestException:
            # Handle network errors or backend unavailability
            return None

    def get_user(self, user_id):
        """
        Get a User object by user_id
        
        :param user_id: Primary key of the user
        :return: User object or None
        """
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None

    def save_tokens(self, user, access_token):
        """
        Save JWT tokens securely
        
        :param user: User object
        :param access_token: JWT access token
        :param refresh_token: JWT refresh token
        """
        # Option 1: Using Django's cache (recommended for temporary storage)
        cache_key = f'user_{user.id}_jwt_tokens'
        cache.set(cache_key, {
            'access_token': access_token
        }, timeout=None)  # No expiration, adjust as needed

        # Option 2: If you want to persist tokens in the database, 
        # you'll need to add custom fields to your User model
        # Uncomment and modify as per your model
        # user.access_token = access_token
        # user.refresh_token = refresh_token
        # user.save()

    def get_access_token(self, user):
        """
        Retrieve the access token for a given user
        
        :param user: User object
        :return: Access token or None
        """
        cache_key = f'user_{user.id}_jwt_tokens'
        tokens = cache.get(cache_key)
        return tokens.get('access_token') if tokens else None