from django.db import models

# Create your models here.

class User(models.Model):
  pictures = models.ImageField(upload_to = "media")
