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
import mlxtend
from mlxtend.frequent_patterns import fpgrowth, association_rules, apriori


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




def my_func(df, algo_type):
    # Import the dataset
    
    response = {'error': False}

    try:
        items = (df['0'].unique())

        encoded_vals = []
        for index, row in df.iterrows():
            labels = {}
            uncommons = list(set(items) - set(row))
            commons = list(set(items).intersection(row))
            for uc in uncommons:
                labels[uc] = 0
            for com in commons:
                labels[com] = 1
            encoded_vals.append(labels)

        # FP-Growth module requires a dataframe that has either 0 and 1 or True and False as data
        # we need to One Hot Encode the data.
        ohe_df = pd.DataFrame(encoded_vals)

        if algo_type == 'fp-growth':
            # Applying fp-growth
            freq_items = fpgrowth(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
        elif algo_type == 'apriori':
            freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)

        # Mining Association Rules
        rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

        # Visualizing results
        # Support vs Confidence
        plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
        plt.xlabel('support')
        plt.ylabel('confidence')
        plt.title('Support vs Confidence')
        support_confidence = generate_graph_img(plt)

        # Support vs Lift
        plt.scatter(rules['support'], rules['lift'], alpha=0.5)
        plt.xlabel('support')
        plt.ylabel('lift')
        plt.title('Support vs Lift')
        support_lift = generate_graph_img(plt)

        # Lift vs Confidence
        fit = np.polyfit(rules['lift'], rules['confidence'], 1)
        fit_fn = np.poly1d(fit)
        plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], 
        fit_fn(rules['lift']))
        lift_confidence = generate_graph_img(plt)
        response = {
            'support_confidence': f'data:image/png;base64,{support_confidence}',
            'support_lift': f'data:image/png;base64,{support_lift}',
            'lift_confidence': f'data:image/png;base64,{lift_confidence}',
            'error': False
        }
    except Exception as e:
        response = {
            'error': str(e)
        }
    return response


def thompson_sampling(df):
    N = 10000
    d = 10
    ads_selected = []
    numbers_of_rewards_1 = [0] * d
    numbers_of_rewards_0 = [0] * d
    total_reward = 0
    response = {'error': False}
    try:
        for n in range(0, N):
            ad = 0
            max_random = 0
            for i in range(0, d):
                random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
                if random_beta > max_random:
                    max_random = random_beta
                    ad = i
            ads_selected.append(ad)
            reward = df.values[n, ad]
            if reward == 1:
                numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
            else:
                numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
            total_reward = total_reward + reward

        # Visualising the results - Histogram
        plt.hist(ads_selected)
        plt.title('Histogram of ads selections')
        plt.xlabel('Ads')
        plt.ylabel('Number of times each ad was selected')
        histogram = generate_graph_img(plt)
        response = {
            'histogram': f'data:image/png;base64,{histogram}',
            'error': False
        }
    except Exception as e:
        response = {
            'error': str(e)
        }
    return response


def upper_confidence_bound(df):
    N = 10000
    d = 10
    ads_selected = []
    numbers_of_selections = [0] * d
    sums_of_rewards = [0] * d
    total_reward = 0
    response = {'error': False}
    try:
        for n in range(0, N):
            ad = 0
            max_upper_bound = 0
            for i in range(0, d):
                if (numbers_of_selections[i] > 0):
                    average_reward = sums_of_rewards[i] / numbers_of_selections[i]
                    delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
                    upper_bound = average_reward + delta_i
                else:
                    upper_bound = 1e400
                if upper_bound > max_upper_bound:
                    max_upper_bound = upper_bound
                    ad = i
            ads_selected.append(ad)
            numbers_of_selections[ad] = numbers_of_selections[ad] + 1
            reward = df.values[n, ad]
            sums_of_rewards[ad] = sums_of_rewards[ad] + reward
            total_reward = total_reward + reward

        # Visualising the results
        plt.hist(ads_selected)
        plt.title('Histogram of ads selections')
        plt.xlabel('Ads')
        plt.ylabel('Number of times each ad was selected')
        histogram = generate_graph_img(plt)
        response = {
            'histogram': f'data:image/png;base64,{histogram}',
            'error': False
        }
    except Exception as e:
        response = {
            'error': str(e)
        }
    return response