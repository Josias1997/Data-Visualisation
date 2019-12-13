from django.shortcuts import get_object_or_404
from data_management_backend.models import File
import pandas as pd
from tabula import read_pdf
import os
import base64
import scipy.stats as st
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, StandardScaler, 
    RobustScaler, MinMaxScaler, Normalizer)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from io import BytesIO
import matplotlib.pyplot as plt


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
            'title': column,
            'field': column,
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
            "title": column_name,
            "field": column_name, 
        },
        {
            "title": function, 
            "field": function
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


def normalize_set(df, function):
    if function == 'std_scaler':
        return StandardScaler().fit_transform(df)
    elif function == 'min_max_scaler':
        return MinMaxScaler().fit_transform(df)
    elif function == 'robust_scaler':
        return RobustScaler().fit_transform(df)
    elif function == 'normalizer':
        return Normalizer().fit_transform(df)


def multi_linear_regression(df, x, y):
    response = {'predict_result': {}, 'error': False}
    try:
        # Multiple Linear Regression
        X = df[[x]]
        y = df[[y]]

        # Encoding categorical data
        labelencoder = LabelEncoder()
        X = labelencoder.fit_transform(X)
        onehotencoder = OneHotEncoder(categories='auto')
        X = onehotencoder.fit_transform(X).toarray()

        # Avoiding the Dummy Variable Trap
        X = X[:, 1:]

        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

        # Feature Scaling
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        sc_y = StandardScaler()
        y_train = sc_y.fit_transform(y_train.reshape(-1, 1))

        # Fitting Multiple Linear Regression to the Training set
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = regressor.predict(X_test)
        print("X_train", X_train)
        print("Y_train", y_train)
        graph_url_train = plot(X_train, y_train, 'blue', 'red', regressor)
        graph_url_test = plot(X_test, y_test, 'blue', 'red', regressor)

        predictions_df = pd.concat([pd.DataFrame(X_test.to_numpy()), pd.DataFrame(y_pred)], axis=1)
        response = {
            'predict_result': predictions_df.to_numpy(),
            'train_plot': f'data:image/png;base64,{graph_url_train}',
            'test_plot': f'data:image/png;base64,{graph_url_test}',
            'error': False,
        }
    except Exception as e:
        response = {
            'error': str(e),
        }
    return response


def linear_regression(df, x, y):
    df = df.select_dtypes(include=['number'])
    independant_value = df[[x]]
    dependant_value = df[[y]]
    response = {'predict_result': {}, 'error': False}
    try:
        X_train, X_test, Y_train, Y_test = train_test_split(independant_value, dependant_value, test_size=0.2)
        regressor = LinearRegression()
        regressor.fit(X_train, Y_train)

        graph_url_train = plot(X_train, Y_train, 'blue', 'red', regressor, x, y, 'Train')
        graph_url_test = plot(X_test, Y_test, 'blue', 'red', regressor, x, y, 'Test')

        predictions_df = pd.concat([pd.DataFrame(X_test.to_numpy()), pd.DataFrame(regressor.predict(X_test))], axis=1)
        response = {
            'predict_result': predictions_df.to_numpy(),
            'train_plot': f'data:image/png;base64,{graph_url_train}',
            'test_plot': f'data:image/png;base64,{graph_url_test}',
            'error': False,
        }
    except Exception as e:
        response = {
            'error': str(e),
        }
    return response


def plot(x, y, first_color, second_color, regressor, x_label=None, y_label=None, type=None):
    plt.scatter(x, y, color=first_color)
    plt.plot(x, regressor.predict(x), color=second_color)
    if x_label != None and y_label != None and type != None:
        plt.title(f'{x_label} vs {y_label} ({type} set)')
        plt.xlabel(f'{x_label}')
        plt.ylabel(f'{y_label}')
    img = BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.clf()
    return graph_url



