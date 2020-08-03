from django.shortcuts import render
import sys
from subprocess import run,PIPE
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
def button(request):
    return render(request,'home.html')
def external(request):
    
    video= run([sys.executable,'C://Users//ishit//Downloads//sih2020_model.py'],shell=False,stdout=PIPE,encoding='utf-8')
    print(video.stdout)
    return render(request,'home.html',{'edit_url':video.stdout})
    print(video.stdout)
    return render(request,'home.html',{'edit_url':video.stdout})