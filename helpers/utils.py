from django.shortcuts import get_object_or_404
from data_management_backend.models import File
import pandas as pd
from tabula import read_pdf
import os


def dataframe_from_file(file):
    pd.set_option('colheader_justify', 'center')
    name, ext = os.path.splitext(file.name)
    if ext == '.csv':
        return pd.read_csv(file, encoding='latin1')
    elif ext == '.pdf':
        return read_pdf(file)
    elif ext == '.txt':
        return pd.read_csv(file, sep='\t', encoding='latin1')
    elif ext == '.json':
        return pd.read_json(file, encoding='latin1')
    elif ext == '.xls' or ext == '.xlsx' or ext == '.xlsm':
        return pd.read_excel(file, encoding='latin1')
    elif ext == '.zip':
        return pd.read_csv(file, compression='zip')
    elif ext == '.gz':
        return pd.read_csv(file, compression='gz')
    elif ext == 'xz':
        return pd.read_csv(file, compression='xz')
    elif ext == '.sav':
        return pd.read_spss(file)
    else:
        return []


def format_to_json(df):
    columns = []
    rows = []
    for column in df.columns.values.tolist():
        column_details = {
            'name': column,
            'label': column,
            'options': {
                'filter': True,
                'sort': True
            }
        }
        columns.append(column_details)

    for index, row in df.iterrows():
        row_details = {}
        for column in df.columns.values.tolist():
            row_details[column] = row[column]
        rows.append(row_details)

    json_object = {'columns': columns, 'rows': rows}
    return json_object


def get_file(pk):
    return get_object_or_404(File, id=pk)
