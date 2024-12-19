from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from .models import ImageUpload

# Custom User Admin to add more functionality
class CustomUserAdmin(UserAdmin):
    list_display = UserAdmin.list_display + ('is_active', 'date_joined')
    list_filter = UserAdmin.list_filter + ('is_active',)
    
    # Custom actions for bulk user management
    actions = ['activate_users', 'deactivate_users']
    
    def activate_users(self, request, queryset):
        queryset.update(is_active=True)
    activate_users.short_description = "Activate selected users"
    
    def deactivate_users(self, request, queryset):
        queryset.update(is_active=False)
    deactivate_users.short_description = "Deactivate selected users"

# ImageUpload Admin Configuration
@admin.register(ImageUpload)
class ImageUploadAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'file_name', 'uploaded_at']
    list_filter = ['user', 'uploaded_at']
    search_fields = ['user__username', 'file_name']
    readonly_fields = ['uploaded_at', 'image_preview']

    def image_preview(self, obj):
        """
        Display a preview of the base64 encoded image in admin
        """
        if obj.image_base64:
            return f'<img src="data:image/jpeg;base64,{obj.image_base64}" style="max-width:300px; max-height:300px;" />'
        return 'No image'
    
    image_preview.short_description = 'Image Preview'
    image_preview.allow_tags = True

# Unregister the default User admin and register custom one
admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)

# Customize admin site headers
admin.site.site_header = "ML Image Annotation Admin"
admin.site.site_title = "ML Image Annotation Portal"
admin.site.index_title = "Welcome to ML Image Annotation Administration"