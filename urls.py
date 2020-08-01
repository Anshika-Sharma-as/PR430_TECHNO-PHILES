from django.contrib import admin
from django.urls import path , include
from uploadcontent import views


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('upload', views.uploadImage , name = 'uploadImage'),
]
