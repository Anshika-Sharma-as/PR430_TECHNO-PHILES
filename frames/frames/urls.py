from django.contrib import admin
from django.urls import path , include
from framesapp import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.button, name='button'),
    path('external', views.external,name='script'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)



