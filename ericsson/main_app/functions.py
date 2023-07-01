import csv
import os

import pandas as pd


def read_file(file):
    if is_csv(file.name):
        f = pd.read_csv(file)
    elif is_excel(file.name):
        f = pd.read_excel(file)
    else:
        raise Exception('File extension not supported')
    return f


def is_csv(filename):
    ext = os.path.splitext(filename)[1]
    return ext == '.csv'


def is_excel(filename):
    ext = os.path.splitext(filename)[1]
    return ext in ['.xls', '.xlsx']
