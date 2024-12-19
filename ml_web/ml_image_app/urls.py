from django.contrib import admin
from django.urls import path
from ml_image_app import views

urlpatterns = [
    path('', views.homepage, name='home'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('upload/', views.upload_image, name='upload'),
    path('train-model/', views.train_model_view, name='train_model'),
    path('my-images/', views.list_images_view, name='image_list'),
    path('my-folders/', views.folders_list_view, name='folder_list'),
    path('folder/<str:folder_id>/', views.folder_detail_view, name='folder_detail'),
    path('save_token/', views.save_token, name='save_token'),
    path('predict/', views.predict_view, name='predict'),
    path('save_bbox/', views.save_bboxes_view, name='save_bbox'),
]