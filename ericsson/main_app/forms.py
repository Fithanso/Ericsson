from django import forms

from .validators import validate_file_extension


class DataUploadForm(forms.Form):
    file = forms.FileField(validators=[validate_file_extension])
