import requests
from django.shortcuts import render
import sys
from subprocess import run,PIPE
from django.core.files.storage import FileSystemStorage
def button(request):
    return render(request,'home.html')
def external(request):
    video=request.FILES['video']
    print("video is ",video)
    fs=FileSystemStorage()
    filename=fs.save(video.name,video)
    fileurl=fs.open(filename)
    templateurl=fs.url(filename)
    print("file raw url",filename)
    print("file full url", fileurl)
    print("template url",templateurl)
    video= run([sys.executable,'C://Users//ishit//Downloads//sih2020_model.py',str(fileurl),str(filename)],shell=False,stdout=PIPE,encoding='utf-8')
    print(video.stdout)
    return render(request,'home.html',{'raw_url':templateurl,'edit_url':video.stdout})
