from django.shortcuts import render

# Create your views here.

def index(request):
    return render( request, 'index.html')


def uploadImage(request):
    print("Request Handling.........")
    p = request.FILES['image']
    from . models import User
    user = User(pictures = p)
    user.save()
    return render( request, 'index.html')

    