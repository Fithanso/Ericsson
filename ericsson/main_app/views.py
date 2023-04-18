from django.http import HttpResponse
from django.shortcuts import render, redirect, reverse
from django.core.files.storage import default_storage
import pandas as pd

from .forms import DataUploadForm
from .functions import *


def index(request):
    form = DataUploadForm()

    if request.method == 'POST':
        form = DataUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            upload_path = f'uploads/first_stage/{file.name}'
            default_storage.save(upload_path, file)
            request.session['first_stage_upload_filepath'] = upload_path
        return redirect(reverse('main_app:first-stage'))
    return render(request, 'main_app/main_page.html', {'form': form})


def first_stage(request):
    file = default_storage.open(request.session.get('first_stage_upload_filepath'))
    try:
        f = read_file(file)
    except:
        return HttpResponse('Failed to open file. Try another file or load your file again.')

    print(f)

    return HttpResponse('result')
