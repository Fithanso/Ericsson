import os
from django.core.exceptions import ValidationError
from main_app.constants import *


def validate_file_extension(value):
    ext = os.path.splitext(value.name)[1]
    valid_extensions = UPLOAD_FILES_EXTENSIONS
    if not ext.lower() in valid_extensions:
        raise ValidationError('Unsupported file extension.')
