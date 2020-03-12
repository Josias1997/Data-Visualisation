from django.shortcuts import get_object_or_404
from data_management_backend.models import File
import pandas as pd
from tabula import read_pdf
import os
import base64
import scipy.stats as st
import scipy.cluster.hierarchy as sch
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, roc_auc_score, adjusted_rand_score,
    homogeneity_score, v_measure_score)
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, StandardScaler, 
    RobustScaler, MinMaxScaler, Normalizer)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import squarify


def dataframe_from_file(file):
    pd.set_option('colheader_justify', 'center')
    name, ext = os.path.splitext(file.name)
    if ext == '.csv':
        return pd.read_csv(file, na_filter=False, encoding='latin1')
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


def plots(kind):
    if kind == 'correlogram':
        ########## Correlogram ####################
        df = pd.read_csv("helpers/datasets/mtcars.csv")

        # Plot cmap color = "RdYlGn", "PRGn", "seismic", "winter", "cool", "bone", "coolwarm", "Wistia", "hot"
        plt.figure(figsize=(12,10), dpi= 80)
        sns.heatmap(df.corr(),
                    cmap='summer',
                    center=0,
                    annot=True)

        # Decorations
        plt.title('Correlogram of mtcars', fontsize=22)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        return generate_graph_img(plt)

    ######### Density plot #####################
    elif kind == 'density-plot':
        df = pd.read_csv("helpers/datasets/mpg_ggplot2.csv")


        # Plot 1 : Density plot
        # Draw Plot
        plt.figure(figsize=(16,10), dpi= 80)
        plt.subplot(211)
        sns.kdeplot(df.loc[df['cyl'] == 4, "cty"], shade=True, color="g", label="Cyl=4", alpha=.7)
        sns.kdeplot(df.loc[df['cyl'] == 5, "cty"], shade=True, color="deeppink", label="Cyl=5", alpha=.7)
        sns.kdeplot(df.loc[df['cyl'] == 6, "cty"], shade=True, color="dodgerblue", label="Cyl=6", alpha=.7)
        sns.kdeplot(df.loc[df['cyl'] == 8, "cty"], shade=True, color="orange", label="Cyl=8", alpha=.7)
        # Decoration
        plt.title('Density Plot', fontsize=22)
        plt.legend()

        plt.subplot(212)
        sns.distplot(df.loc[df['class'] == 'compact', "cty"], color="dodgerblue", label="Compact", hist_kws={'alpha':.7}, kde_kws={'linewidth':3})
        sns.distplot(df.loc[df['class'] == 'suv', "cty"], color="orange", label="SUV", hist_kws={'alpha':.7}, kde_kws={'linewidth':3})
        sns.distplot(df.loc[df['class'] == 'minivan', "cty"], color="g", label="minivan", hist_kws={'alpha':.7}, kde_kws={'linewidth':3})
        plt.ylim(0, 0.35)

        # Decoration
        plt.title('Density Plot of City Mileage by Vehicle Type', fontsize=22)
        plt.legend()
        return generate_graph_img(plt)

    ########## Treemap #################
    elif kind == 'treemap':
        # Import Data
        df_raw = pd.read_csv("helpers/datasets/mpg_ggplot2.csv")

        # Prepare Data
        df = df_raw.groupby('class').size().reset_index(name='counts')
        labels = df.apply(lambda x: str(x[0]) + "\n (" + str(x[1]) + ")", axis=1)
        sizes = df['counts'].values.tolist()
        colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]

        # Draw Plot
        plt.figure(figsize=(12,8), dpi= 80)
        squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.8)

        # Decorate
        plt.title('Treemap of Vechile Class')
        plt.axis('off')
        return generate_graph_img(plt)


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
    elif function == 'stdev(x)':
        return np.std(column)

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
        matrix_plot = generate_graph_img(plt)

        # Classificatin report
        report = classification_report(y_test, y_pred)

        # Probabilité des prédictions
        prob_pred = classifier.predict_proba(X_train)
        y_score = prob_pred[:,1]

        # Tracer la courbe ROC

        faux_positive, vrai_positive, seuils = roc_curve(y_train, y_score)
        plt.figure(figsize=(11, 8))
        courbe_roc_func(faux_positive, vrai_positive)
        courbe_roc = generate_graph_img(plt)
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
        graph_url_train = generate_graph_img(plt)

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
        graph_url_test = generate_graph_img(plt)
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
    y = df.iloc[:, 2:3].values

    response = {'error': False}
    try:
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(X)
        y = sc_y.fit_transform(y)

        regressor = SVR(kernel = 'rbf')
        regressor.fit(X, y)

        # Predicting a new result
        y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))
        y_pred = sc_y.inverse_transform(y_pred)

        # Visualising the SVR results
        plt.scatter(X, y, color = 'red')
        plt.plot(X, regressor.predict(X), color = 'blue')
        plt.title('Truth or Bluff (SVR)')
        plt.xlabel('Position level')
        plt.ylabel('Salary')
        svr_results = generate_graph_img(plt)

        # Visualising the SVR results (for higher resolution and smoother curve)
        X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
        X_grid = X_grid.reshape((len(X_grid), 1))
        plt.scatter(X, y, color = 'red')
        plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
        plt.title('Truth or Bluff (SVR)')
        plt.xlabel('Position level')
        plt.ylabel('Salary')
        svr_results_hr = generate_graph_img(plt)
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

def k_nearest_neighbors(df):
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

        # Fitting K-NN to the Training set
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
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
        matrix_plot = generate_graph_img(plt)


        # data = y_train, X_train
        # data=data.reshape((100,300))

        # Probabilité de nos predictions
        prob_pred = classifier.predict_proba(X_train)
        y_score = prob_pred[:,1]

        faux_positive, vrai_positive, seuils = roc_curve(y_train, y_score)
        plt.figure(figsize=(12, 8))
        courbe_roc_func(faux_positive, vrai_positive)
        courbe_roc = generate_graph_img(plt)

        score_roc = roc_auc_score(y_train, y_score)
        score_roc = score_roc * 100


        # Visualising the Training set results
        X_set, y_set = X_train, y_train
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('orange', 'yellow')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('blue', 'grey'))(i), label = j)
        plt.title('K-NN (Training set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        graph_url_train = generate_graph_img(plt)

        # Visualising the Test set results
        X_set, y_set = X_test, y_test
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('yellow', 'orange')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('blue', 'yellow'))(i), label = j)
        plt.title('K-NN (Test set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        graph_url_test = generate_graph_img(plt)

        response = {
            'matrix_plot': f'data:image/png;base64,{matrix_plot}',
            'courbe_roc': f'data:image/png;base64,{courbe_roc}',
            'score_roc': score_roc,
            'train_plot': f'data:image/png;base64,{graph_url_train}',
            'test_plot': f'data:image/png;base64,{graph_url_test}',
            'confusion_matrix': cm,
            'error': False,
        }
    except Exception as e:
        response = {
            'error': str(e)
        }
    return response


def support_vector_machine(df, kernel):
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

        # Fitting SVM to the Training set
        classifier = SVC(kernel = kernel, random_state = 0, probability = True)
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
        matrix_plot = generate_graph_img(plt)

        # Probabilité de nos predictions
        prob_pred = classifier.predict_proba(X_train)
        y_score = prob_pred[:,1]

        # from sklearn.metrics import roc_curve, auc
        # Courbe ROC
        faux_positive, vrai_positive, seuils = roc_curve(y_train, y_score)
        plt.figure(figsize=(12, 8))
        courbe_roc_func(faux_positive, vrai_positive)
        courbe_roc = generate_graph_img(plt)
        score_roc = roc_auc_score(y_train, y_score)
        score_roc = score_roc * 100

        # Visualising the Training set results
        X_set, y_set = X_train, y_train
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('blue', 'yellow')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('red', 'green'))(i), label = j)
        plt.title('SVM (Training set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        graph_url_train = generate_graph_img(plt)

        # Visualising the Test set results
        X_set, y_set = X_test, y_test
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('orange', 'blue')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('red', 'green'))(i), label = j)
        plt.title('SVM (Test set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        graph_url_test = generate_graph_img(plt)
        response = {
            'matrix_plot': f'data:image/png;base64,{matrix_plot}',
            'courbe_roc': f'data:image/png;base64,{courbe_roc}',
            'score_roc': score_roc,
            'train_plot': f'data:image/png;base64,{graph_url_train}',
            'test_plot': f'data:image/png;base64,{graph_url_test}',
            'confusion_matrix': cm,
            'error': False,
        }
    except Exception as e:
        response = {
            'error': str(e)
        }
    return response



def decision_tree_regressor(df):
    X = df.iloc[:, 1:2].values
    y = df.iloc[:, 2].values

    response = {'error': False}
    try:
        # Fitting Decision Tree Regression to the dataset
        regressor = DecisionTreeRegressor(random_state = 0)
        regressor.fit(X, y)

        # Predicting a new result
        y_pred = regressor.predict(X)

        # Visualising the Decision Tree Regression results (higher resolution)
        X_grid = np.arange(min(X), max(X), 0.01)
        X_grid = X_grid.reshape((len(X_grid), 1))
        plt.scatter(X, y, color = 'red')
        plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
        plt.title('Truth or Bluff (Decision Tree Regression)')
        plt.xlabel('Position level')
        plt.ylabel('Salary')
        decision_tree_graph_img = generate_graph_img(plt) 
        response = {
            'decision_tree_graph_img': f'data:image/png;base64,{decision_tree_graph_img}',
            'error': False
        }
    except Exception as e:
        response = {
            'error': str(e)
        }
    return response

def random_forest_regression(df):
    X = df.iloc[:, 1:2].values
    y = df.iloc[:, 2].values

    response = {'error': False}
    try:
        # Fitting Random Forest Regression to the dataset
        regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
        regressor.fit(X, y)

        # Predicting a new result
        y_pred = regressor.predict(X)

        # Visualising the Random Forest Regression results (higher resolution)
        X_grid = np.arange(min(X), max(X), 0.01)
        X_grid = X_grid.reshape((len(X_grid), 1))
        plt.scatter(X, y, color = 'red')
        plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
        plt.title('Truth or Bluff (Random Forest Regression)')
        plt.xlabel('Position level')
        plt.ylabel('Salary')
        rdm_forest_regression_graph = generate_graph_img(plt)
        response = {
            'rdm_forest_regression_graph': f'data:image/png;base64,{rdm_forest_regression_graph}',
            'error': False
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
    graph_url = generate_graph_img(plt)
    return graph_url

def plot_scatter(x, y, first_color, second_color, x_label=None, y_label=None, type=None):
    plt.figure(figsize=(11, 8))
    plt.scatter(x, y, color=first_color)
    if x_label != None and y_label != None and type != None:
        plt.title(f'{x_label} vs {y_label} ({type} set)')
        plt.xlabel(f'{x_label}')
        plt.ylabel(f'{y_label}')
    graph_url = generate_graph_img(plt)
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


def generate_graph_img(plot):
    img = BytesIO()
    plot.savefig(img, format="png")
    img.seek(0)
    graph_img = base64.b64encode(img.getvalue()).decode()
    plot.clf()
    return graph_img


def classification(df, classifier_type, plot_title):
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

        # Fitting Decision Tree Classification to the Training set
        classifier = None
        if classifier_type == 'decision-tree':     
            classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        elif classifier_type == 'naives-bayes':
            classifier = GaussianNB()
        elif classifier_type == 'random-forest':
            classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

        classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)

        # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize = (10,7))
        ax = sns.heatmap(cm, annot=True)
        fig = ax.get_figure()
        img = BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        matrix_plot = base64.b64encode(img.getvalue()).decode()
        plt.clf()
        
        # Classification report
        report = classification_report(y_test, y_pred)

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
                        c = ListedColormap(('orange', 'blue'))(i), label = j)
        plt.title(f'{plot_title} (Training set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        graph_url_train = generate_graph_img(plt)

        # Visualising the Test set results
        X_set, y_set = X_test, y_test
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('blue', 'red')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('orange', 'green'))(i), label = j)
        plt.title(f'{plot_title} (Test set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        graph_url_test = generate_graph_img(plt)

        # Probabilité de nos predictions
        prob_pred = classifier.predict_proba(X_train)
        y_score = prob_pred[:,1]

        # from sklearn.metrics import roc_curve, auc
        # Courbe ROC
        faux_positive, vrai_positive, seuils = roc_curve(y_train, y_score)

        plt.figure(figsize=(12, 8))
        courbe_roc_func(faux_positive, vrai_positive)
        courbe_roc = generate_graph_img(plt)

        score_roc = roc_auc_score(y_train, y_score)
        score_roc = score_roc * 100

        response = {
            'matrix_plot': f'data:image/png;base64,{matrix_plot}',
            'courbe_roc': f'data:image/png;base64,{courbe_roc}',
            'score_roc': score_roc,
            'train_plot': f'data:image/png;base64,{graph_url_train}',
            'test_plot': f'data:image/png;base64,{graph_url_test}',
            'confusion_matrix': cm,
            'error': False,
        }

    except Exception as e:
        response = {
            'error': str(e)
        }
    return response


def k_means_cluster(df):
    X = df.iloc[:, [3, 4]].values
    y = df.iloc[:, 3:4].values
    response = {'error': False}
    try:
        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

        # Feature Scaling
        print("Done")
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        sc_y = StandardScaler()
        y_train = sc_y.fit_transform(y_train)

        # Using the elbow method to find the optimal number of clusters 
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, 11), wcss)
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        elbow_graph = generate_graph_img(plt)

        # Fitting K-Means to the dataset
        Kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)

        # Predic
        y_kmeans = kmeans.fit_predict(X)
        rand_score = adjusted_rand_score(y.ravel(), y_kmeans)

        # Visualising the clusters
        plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
        plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
        plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
        plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
        plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
        plt.title('Clusters of customers')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        clusters = generate_graph_img(plt)
        response = {
            'elbow_graph': f'data:image/png;base64,{elbow_graph}',
            'clusters': f'data:image/png;base64,{clusters}',
            'error': False
        }
    except Exception as e:
        response = {
            'error': str(e)
        }
    return response


def hierarchical_cluster(df):
    X = df.iloc[:, [3, 4]].values
    y = df.iloc[:, 3:4].values
    response = {'error': False}
    try:
        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

        # Feature Scaling
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        sc_y = StandardScaler()
        y_train = sc_y.fit_transform(y_train)

        # Using the dendrogram to find the optimal number of clusters
        dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
        plt.title('Dendrogram')
        plt.xlabel('Customers')
        plt.ylabel('Euclidean distances')
        dendrogram_graph = generate_graph_img(plt)

        # Fitting Hierarchical Clustering to the dataset
        hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

        # predict
        y_hc = hc.fit_predict(X)

        rand_score = adjusted_rand_score(y.ravel(), y_hc)
        homogeneity_score(y.ravel(), y_hc)
        v_measure_score(y.ravel(), y_hc)

        # Visualising the clusters
        plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
        plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
        plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'orange', label = 'Cluster 3')
        plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'grey', label = 'Cluster 4')
        plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'yellow', label = 'Cluster 5')
        plt.title('Clusters of customers')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        clusters = generate_graph_img(plt)
        response = {
            'dendrogram': f'data:image/png;base64,{dendrogram_graph}',
            'clusters': f'data:image/png;base64,{clusters}',
            'error': False
        }
    except Exception as e:
        response = {
            'error': str(e)
        }
    return response


def lda(df):
    X = df.iloc[:, 0:13].values
    y = df.iloc[:, 13].values

    response = {'error': False}

    try:
        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Applying LDA
        lda = LDA(n_components = 2)
        X_train = lda.fit_transform(X_train, y_train)
        X_test = lda.transform(X_test)

        # Fitting Logistic Regression to the Training set
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)

        # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize = (10,7))
        ax = sns.heatmap(cm, annot=True)
        fig = ax.get_figure()
        img = BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        matrix_plot = base64.b64encode(img.getvalue()).decode()
        plt.clf()

        report = classification_report(y_test, y_pred)

        # Visualising the Training set results
        X_set, y_set = X_train, y_train
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('lightblue', 'orange', 'blue')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('red', 'green', 'orange'))(i), label = j)
        plt.title('LDA (Training set)')
        plt.xlabel('LD1')
        plt.ylabel('LD2')
        plt.legend()
        graph_url_train = generate_graph_img(plt)

        # Visualising the Test set results
        X_set, y_set = X_test, y_test
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('blue', 'grey', 'lightblue')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
        plt.title('LDA (Test set)')
        plt.xlabel('LD1')
        plt.ylabel('LD2')
        plt.legend()
        graph_url_test = generate_graph_img(plt)
        response = {
            'matrix_plot': f'data:image/png;base64,{matrix_plot}',
            'train_plot': f'data:image/png;base64,{graph_url_train}',
            'test_plot': f'data:image/png;base64,{graph_url_test}',
            'confusion_matrix': cm,
            'error': False,
        }
    except Exception as e:
        response = {
            'error': str(e)
        }
    return response


def pca(df):
    X = df.iloc[:, 0:13].values
    y = df.iloc[:, 13].values

    response = {'error': False}
    try:
        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Applying PCA
        pca = PCA(n_components = 2)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        explained_variance = pca.explained_variance_ratio_

        # Fitting Logistic Regression to the Training set
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)

        # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize = (10,7))
        ax = sns.heatmap(cm, annot=True)
        fig = ax.get_figure()
        img = BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        matrix_plot = base64.b64encode(img.getvalue()).decode()
        plt.clf()

        report = classification_report(y_test, y_pred)

        # Visualising the Training set results
        X_set, y_set = X_train, y_train
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('orange', 'grey', 'blue')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
        plt.title('PCA (Training set)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        graph_url_train = generate_graph_img(plt)

        # Visualising the Test set results
        X_set, y_set = X_test, y_test
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('orange', 'grey', 'blue')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
        plt.title('PCA (Test set)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        graph_url_test = generate_graph_img(plt)
        response = {
            'matrix_plot': f'data:image/png;base64,{matrix_plot}',
            'train_plot': f'data:image/png;base64,{graph_url_train}',
            'test_plot': f'data:image/png;base64,{graph_url_test}',
            'confusion_matrix': cm,
            'error': False,
        }
    except Exception as e:
        response = {
            'error': str(e)
        }

    return response


def kpca(df):
    X = df.iloc[:, [2, 3]].values
    y = df.iloc[:, 4].values

    try:
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Applying Kernel PCA
        kpca = KernelPCA(n_components = 2, kernel = 'rbf')
        X_train = kpca.fit_transform(X_train)
        X_test = kpca.transform(X_test)

        # Fitting Logistic Regression to the Training set
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)

        # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize = (10,7))
        ax = sns.heatmap(cm, annot=True)
        fig = ax.get_figure()
        img = BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        matrix_plot = base64.b64encode(img.getvalue()).decode()

        report = classification_report(y_test, y_pred)

        # Visualising the Training set results
        X_set, y_set = X_train, y_train
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('pink', 'red')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('black', 'white'))(i), label = j)
        plt.title('Kernel PCA (Training set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        graph_url_train = generate_graph_img(plt)

        # Visualising the Test set results
        X_set, y_set = X_test, y_test
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('blue', 'purple')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('red', 'black'))(i), label = j)
        plt.title('Kernel PCA (Test set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        graph_url_test = generate_graph_img(plt)
        response = {
            'matrix_plot': f'data:image/png;base64,{matrix_plot}',
            'train_plot': f'data:image/png;base64,{graph_url_train}',
            'test_plot': f'data:image/png;base64,{graph_url_test}',
            'confusion_matrix': cm,
            'error': False,
        }
    except Exception as e:
        response = {
            'error': str(e)
        }
    return response




