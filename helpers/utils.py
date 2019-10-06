import pandas as pd
import os


def dataframe_from_file(file):
    pd.set_option('colheader_justify', 'center')
    name, ext = os.path.splitext(file.name)
    if ext == '.csv':
        return pd.read_csv(file, encoding='latin1')
    elif ext == '.json':
        return pd.read_json(file, encoding='latin1')
    elif ext == '.xls' or ext == '.xlsx' or ext == '.xlsm':
        return pd.read_excel(file, encoding='latin1')
    else:
        return []


def style_dataframe(df):
    return df.style.set_properties(**{
        'font-size': '11pt',
        'font-family': 'Arial',
        'font-weight': 'bold',
        'background-color': 'black',
        'color': 'lawngreen'
    })

