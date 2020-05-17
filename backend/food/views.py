from django.shortcuts import render
from django.http import HttpResponse
from .predict import predict
from django.core.files.storage import FileSystemStorage

def home(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        fs.save(uploaded_file.name, uploaded_file)
        # predict(uploaded_file.name)

    return render(request, 'food/predict.html')

def about(request):
    return HttpResponse('<h1>about</h1>')
