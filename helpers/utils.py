from django.shortcuts import get_object_or_404
from data_management_backend.models import File
import pandas as pd
from tabula import read_pdf
import os
import scipy.stats as st
import numpy as np


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


def compute_stats(x, y, test):
    if test == 'normtest':
        return {
            "normaltest": st.normaltest(x),
            "shapiro": st.shapiro(x)
        }
    elif test == 'skewtest':
        return st.skewtest(x)
    elif test == 'cumfreq':
        freq = st.cumfreq(x)
        return {
            "cumcount": freq.cumcount,
            "lowerlimit": freq.lowerlimit,
            "binsize": freq.binsize,
            "extrapoints": freq.extrapoints
        }
    elif test == 'correlation':
        return {
            "pearsonr": st.pearsonr(x, y),
            "spearmanr": st.spearmanr(x, y)
        }
    elif test == 't-test':
        return st.ttest_ind(x, y)
    elif test == 'anova':
        return st.f_oneway(x, y)
    elif test == 'chisquare':
        return st.chisquare(x)
    elif test == 'fisher_exact':
        return st.fisher_exact([[x[0], x[1]], [y[0], y[1]]])
    elif test == 'wilcoxon':
        return st.wilcoxon(x, y)
    elif test == 'zscore':
        return st.zscore(x)


def call_math_function(function, column):
    if function == 'log(x)':
        return np.log(column, where=(column!=0))
    elif function == 'log10(x)':
        return np.log10(column, where=(column!=0))
    elif function == 'abs(x)':
        return np.absolute(column)
    elif function == 'median(x)':
        return np.median(column)
    elif function == 'quantile(x)':
        return np.quantile(column, 0.5)
    elif function == 'exp(x)':
        return np.exp(column)
    elif function == 'round(x)':
        return np.ndarray.round(column)
    elif function == 'signif(x)':
        return np.ceil(column)
    elif function == 'sin(x)':
        return np.sin(column)
    elif function == 'cos(x)':
        return np.cos(column)
    elif function == 'tan(x)':
        return np.tan(column)
    elif function == 'max(x)':
        return np.ndarray.max(column)
    elif function == 'min(x)':
        return np.ndarray.min(column)
    elif function == 'length(x)':
        return np.array([column.size])
    elif function == 'range(x)':
        return np.array([np.ndarray.min(column), np.ndarray.max(column)])
    elif function == 'sum(x)':
        return np.sum(column)
    elif function == 'prod(x)':
        return np.prod(column)
    elif function == 'mean(x)':
        return np.mean(column)
    elif function == 'var(x)':
        return np.var(column)
    elif function == 'sqrt(x)':
        return np.sqrt(column)
    elif function == 'sort(x)':
        return np.sort(column)

def format_np_array(array, function, column, column_name):
    columns = [
        {
            "name": column_name,
            "label": column_name, 
        },
        {
            "name": function, 
            "label": function
        }
    ]
    rows = []
    index = 0
    print(array)
    for value in np.nditer(column):
        if index < array.size:
            rows.append({
                column_name: value,
                function: array[index]
            })
        else:
            rows.append({
                column_name: value,
                function: "...."
            })
        index = index + 1
    return {'columns': columns, 'rows': rows}
