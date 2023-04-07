from django.http import HttpResponse
from django.shortcuts import render

from .forms import DataUploadForm


def index(request):
    form = DataUploadForm()
    return render(request, 'templates/main_page.html', {'form': form})
