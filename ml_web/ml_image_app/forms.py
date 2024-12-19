from django import forms
from .models import ImageUpload
from django.core.exceptions import ValidationError
from django.core.validators import FileExtensionValidator
# class ImageUploadForm(forms.Form):
#     image = forms.CharField(widget=forms.HiddenInput())

#     def save(self, user=None, commit=True):
#         """
#         Custom save method to handle base64 conversion
#         """
#         # Get the base64 image string
#         image_base64 = self.cleaned_data.get('image')

#         # Create model instance
#         image_upload = ImageUpload(user=user)

#         # Save base64 image
#         image_upload.save_base64_image_from_data(image_base64)
        
#         return image_upload

def validate_file_size(value):
    filesize = value.size
    
    if filesize > 42 * 1024 * 1024:  # 42 megabytes
        raise ValidationError("The maximum file size that can be uploaded is 42 megabytes")
    
class ImageUploadForm(forms.Form):
    image = forms.FileField(
        label='Select an image',
        help_text='Max. 42 megabytes',
        validators=[
            FileExtensionValidator(
                allowed_extensions=['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'],
                message='Only image files are allowed.'
            ),
            validate_file_size
        ]
    )

    def clean_image(self):
        image = self.cleaned_data.get('image', False)
        
        if image:
            # Additional validation can be added here if needed
            return image
        
        raise forms.ValidationError("Couldn't read uploaded image.")
    
class BulkImageUploadForm(forms.Form):
    images = forms.FileField(
        label='Select images',
        help_text='Max. 42 megabytes per image',
        widget=forms.FileInput(attrs={'allow_multiple_selected': True}),
        validators=[
            FileExtensionValidator(
                allowed_extensions=['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'],
                message='Only image files are allowed.'
            )
        ]
    )

    def clean_images(self):
        images = self.cleaned_data.get('images', [])
        uploaded_files = self.files.getlist('images')
        
        if not uploaded_files:
            raise forms.ValidationError("No images selected.")
        
        # Validate file size for each image
        for image in uploaded_files:
            if image.size > 42 * 1024 * 1024:  # 42 megabytes
                raise ValidationError(f"The file {image.name} exceeds the maximum file size of 42 megabytes")
        
        return uploaded_files