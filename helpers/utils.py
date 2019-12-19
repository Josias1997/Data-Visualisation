from django.shortcuts import get_object_or_404
from data_management_backend.models import File
import pandas as pd
from tabula import read_pdf
import os
import base64
import scipy.stats as st
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.svm import SVR
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, StandardScaler, 
    RobustScaler, MinMaxScaler, Normalizer)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns


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


def multi_linear_regression(df):
    response = {'predict_result': {}, 'error': False}
    try:
        # Multiple Linear Regression
        X = df.iloc[:, :-1].values
        y = df.iloc[:, 4].values

        # Encoding categorical data
        labelencoder = LabelEncoder()
        X[:, 3] = labelencoder.fit_transform(X[:, 3])
        onehotencoder = OneHotEncoder(categories='auto')
        X = onehotencoder.fit_transform(X).toarray()

        # Avoiding the Dummy Variable Trap
        X = X[:, 1:]
        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

        # Feature Scaling
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(pd.DataFrame(X_test))
        sc_y = StandardScaler()

        y_train = sc_y.fit_transform(pd.DataFrame(y_train))

        # Fitting Multiple Linear Regression to the Training set
        regressor = LinearRegression()
        regressor.fit(pd.DataFrame(X_train), pd.DataFrame(y_train))

        # Predicting the Test set results
        y_pred = regressor.predict(X_test)
        print("PLOT")
        graph_url = seaborn_plot(y_train, y_pred)
        administration = df[['Administration']]
        rd_spend = df[['R&D Spend']]
        marketing_spend = df[['Marketing Spend']]
        state = df[['State']]
        profit = df[['Profit']]
        plot_admin = plot_scatter(administration, profit, 'blue', 'red', 'Administration', 'Profit', 'Administration & Profit')
        plot_rd_spend = plot_scatter(rd_spend, profit, 'blue', 'red', 'R&D Spend', 'Profit', 'R&D Spend & Profit')
        plot_marketing_spend = plot_scatter(marketing_spend, profit, 'blue', 'red', 'Marketing Spend', 'Profit', 'Marketing Spend & Profit')
        # plot_state = plot_scatter(state, profit, 'blue', 'red', 'State', 'Profit', 'State & Profit')
        response = {
            'predict_result': [],
            'seaborn_plot': f'data:image/png;base64,{graph_url}',
            'admin_plot': f'data:image/png;base64,{plot_admin}',
            'rd_spend_plot': f'data:image/png;base64,{plot_rd_spend}',
            'marketing_plot': f'data:image/png;base64,{plot_marketing_spend}',
            'error': False,
        }
    except Exception as e:
        response = {
            'predict_result': [],
            'error': str(e),
        }
    return response


def logistic_regression(df):
    X = df.iloc[:, [2, 3]].values
    y = df.iloc[:, 4].values
    response = {'error': False}
    try:

        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Fitting classifier to the Training set
        # Create your classifier here
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)

        # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.matshow(cm)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        img = BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        matrix_plot = base64.b64encode(img.getvalue()).decode()
        plt.clf()

        # Classificatin report
        report = classification_report(y_test, y_pred)
        print(report)

        # Probabilité des prédictions
        prob_pred = classifier.predict_proba(X_train)
        y_score = prob_pred[:,1]

        # Tracer la courbe ROC

        faux_positive, vrai_positive, seuils = roc_curve(y_train, y_score)
        plt.figure(figsize=(11, 8))
        courbe_roc_func(faux_positive, vrai_positive)
        img = BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        courbe_roc = base64.b64encode(img.getvalue()).decode()
        plt.clf()
        score_roc = roc_auc_score(y_train, y_score)
        score_roc = score_roc * 100



        # Visualising the Training set results
        X_set, y_set = X_train, y_train
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('red', 'green'))(i), label = j)
        plt.title('Classifier (Training set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        img = BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        graph_url_train = base64.b64encode(img.getvalue()).decode()
        plt.clf()

        # Visualising the Test set results
        X_set, y_set = X_test, y_test
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('red', 'green'))(i), label = j)
        plt.title('Classifier (Test set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        img = BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        graph_url_test = base64.b64encode(img.getvalue()).decode()
        plt.clf()
        response = {
            'matrix_plot': f'data:image/png;base64,{matrix_plot}',
            'report': report,
            'courbe_roc': f'data:image/png;base64,{courbe_roc}',
            'score_roc': score_roc,
            'train_plot': f'data:image/png;base64,{graph_url_train}',
            'test_plot': f'data:image/png;base64,{graph_url_test}',
            'confusion_matrix': cm,
            'error': False,
        }
    except Exception as e:
        response = {
            'error': str(e),
        }
    return response

def courbe_roc_func(faux_positive, vrai_positive, label=None):
    plt.plot(faux_positive, vrai_positive, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('faux positif', fontsize=18)
    plt.ylabel('vrai positif', fontsize=18)


def svr(df):
    X = df.iloc[:, 1:2].values
    y = df.iloc[:, 2].values

    # Splitting the dataset into the Training set and Test set
    """from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""
    response = {'error': False}
    try:
        # Feature Scaling
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(X)
        y = sc_y.fit_transform([y])

        regressor = SVR(kernel = 'rbf')
        regressor.fit(X, y.ravel())

        # Predicting a new result
        y_pred = regressor.predict(X)
        y_pred = sc_y.inverse_transform(y_pred)

        # Visualising the SVR results
        plt.scatter(X, y, color = 'red')
        plt.plot(X, regressor.predict(X), color = 'blue')
        plt.title('Truth or Bluff (SVR)')
        plt.xlabel('Position level')
        plt.ylabel('Salary')
        img = BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        svr_results = base64.b64encode(img.getvalue()).decode()
        plt.clf()

        # Visualising the SVR results (for higher resolution and smoother curve)
        X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
        X_grid = X_grid.reshape((len(X_grid), 1))
        plt.scatter(X, y, color = 'red')
        plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
        plt.title('Truth or Bluff (SVR)')
        plt.xlabel('Position level')
        plt.ylabel('Salary')
        img = BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        svr_results_hr = base64.b64encode(img.getvalue()).decode()
        plt.clf()
        response = {
            'svr_results': f'data:image/png;base64,{svr_results}',
            'svr_results_hr': f'data:image/png;base64,{svr_results_hr}',
            'error': False,
        }
    except Exception as e:
        response = {
            'error': str(e)
        }
    return response


def linear_regression(df, x, y):
    df = df.select_dtypes(include=['number'])
    independant_value = df[[x]]
    dependant_value = df[[y]]
    response = {'predict_result': [], 'error': False}
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
            'predict_result': [],
            'error': str(e),
        }
    return response


def plot(x, y, first_color, second_color, regressor, x_label=None, y_label=None, type=None):
    plt.figure(figsize=(11, 8))
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

def plot_scatter(x, y, first_color, second_color, x_label=None, y_label=None, type=None):
    plt.figure(figsize=(11, 8))
    plt.scatter(x, y, color=first_color)
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

def seaborn_plot(y1, y2):
    ax1 = sns.distplot(y1, hist=False, color="r", label="valeurs réelles")
    sns.distplot(y2, hist=False, color="b", label="valeurs prévues" , ax=ax1)
    fig = ax1.get_figure()
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.clf()
    return graph_url



