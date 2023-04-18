from django.urls import path

from .views import *

app_name = 'main_app'

urlpatterns = [
    path('', index, name='index'),
    path('first-stage', first_stage, name='first-stage')
]
