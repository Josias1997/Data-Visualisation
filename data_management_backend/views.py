from django.shortcuts import render, get_object_or_404, HttpResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from .models import File, Token, PlotFile
from django.contrib.auth.models import User
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.http import Http404
from pandasql import sqldf
from django.conf import settings
from decimal import Decimal
from io import BytesIO
from helpers.utils import (dataframe_from_file, plots, 
    format_to_json, compute_stats, call_math_function, 
    format_np_array, normalize_set, decision_tree_regressor, random_forest_regression,
    multi_linear_regression, linear_regression, logistic_regression, lda, pca, kpca,
    svr, k_nearest_neighbors, support_vector_machine, classification, k_means_cluster, hierarchical_cluster)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import base64
import scipy.stats
import pandas as pd
import os
from helpers.deep_learning import (my_func, thompson_sampling, upper_confidence_bound, artificial_neural_network,
    recurrent_neural_network, convolutional_neural_network)
from helpers.text_mining import sentimental_analysis
import seaborn as sns
import missingno as msno

# Create your views here.

@api_view(http_method_names=['POST'])
@parser_classes([MultiPartParser])
def upload(request):
    file = File(file=request.data['file'])
    name, ext = os.path.splitext(file.file.name)
    file.title = name
    file.save()
    df = dataframe_from_file(file.file)
    json_response = {
        'data': 'Extension non valide'
    }
    if len(df) != 0:
        json_data = format_to_json(df)
        columns = df.columns.values.tolist()
        json_response = {
            'name': name,
            'data': json_data,
            'path': file.file.url,
            'columnsNames': columns,
            'id': file.id,
            'size': df.size,
            'rows': df.shape[0],
            'columns': df.shape[1],
        }
    else:
        file.delete()
    return Response(json_response)


@api_view(http_method_names=['POST'])
def filter_by_columns_and_rows(request):
    pk = request.data['id']
    file = get_object_or_404(File, id=pk)
    df = dataframe_from_file(file.file)
    begin_line = int(request.data['beginLine']) if request.data['beginLine'] else ''
    end_line = int(request.data['endLine']) if request.data['endLine'] else ''
    begin_column = int(request.data['beginColumn']) if request.data['beginColumn'] else ''
    end_column = int(request.data['endColumn']) if request.data['endColumn'] else ''
    last_line = df.shape[0] + 1
    last_column = df.shape[1] + 1
    filtered_data = df
    if (begin_line != '' and (begin_line < 0 or begin_line >= last_line - 1)) or (
            end_line != '' and end_line > last_line) or (
            begin_column != '' and (begin_column < 0 or begin_column >= last_column - 1)) or (
            end_column != '' and end_column > last_column):
        return HttpResponse("Indices invalides (Index lignes ou colonnes en dehors de la plage autorisée)")
    if begin_line != '' and end_line != '' and begin_column != '' and end_column != '':
        filtered_data = df.iloc[begin_line:end_line, begin_column:end_column]
    elif begin_line == '' and end_line != '' and begin_column != '' and end_column != '':
        filtered_data = df.iloc[:end_line, begin_column:end_column]
    elif begin_line == '' and end_line == '' and begin_column != '' and end_column != '':
        filtered_data = df.iloc[:, begin_column:end_column]
    elif begin_line == '' and end_line == '' and begin_column == '' and end_column != '':
        filtered_data = df.iloc[:, :end_column]
    elif begin_line != '' and end_line == '' and begin_column == '' and end_column == '':
        filtered_data = df.iloc[begin_line:, :]
    elif begin_line != '' and end_line != '' and begin_column == '' and end_column == '':
        filtered_data = df.iloc[begin_line:end_line, :]
    elif begin_line != '' and end_line != '' and begin_column != '' and end_column == '':
        filtered_data = df.iloc[begin_line:end_line, begin_column:]
    elif begin_line != '' and end_line == '' and begin_column == '' and end_column != '':
        filtered_data = df.iloc[begin_line:, :end_column]
    elif begin_line != '' and end_line != '' and begin_column == '' and end_column != '':
        filtered_data = df.iloc[begin_line:end_line, :end_column]
    elif begin_line != '' and end_line == '' and begin_column != '' and end_column != '':
        filtered_data = df.iloc[begin_line:, begin_column:end_column]
    elif begin_line == '' and end_line != '' and begin_column != '' and end_column == '':
        filtered_data = df.iloc[:end_line, begin_column:]
    elif begin_line != '' and end_line == '' and begin_column != '' and end_column == '':
        filtered_data = df.iloc[begin_line:, begin_column:]
    elif begin_line == '' and end_line != '' and begin_column == '' and end_column != '':
        filtered_data = df.iloc[:end_line, :end_column]
    return HttpResponse(format_to_json(filtered_data))


@api_view(http_method_names=['POST'])
def login(request):
    username = request.data['username']
    password = request.data['password']
    try:
        user = User.objects.get(username=username)
        if not user.check_password(password):
            raise ValidationError('Invalid Password', code='invalid')
    except ObjectDoesNotExist:
        raise Http404
    except ValidationError:
        raise Http404
    token = Token()
    token.user = user
    token.generateToken()
    token.save()
    return Response({
        'key': token.string,
    })



@api_view(http_method_names=['POST'])
def sign_up(request):
    username = request.data['username']
    password = request.data['password']
    email = request.data['email']
    try:
        user = User.objects.get(username=username)
        raise ValidationError('Username is already used', code='invalid')
    except ObjectDoesNotExist:
        User.objects.create_user(username, email, password)
        return Response({
                'username': username,
                'password': password,
                'status': 'success',
            })
    except ValidationError:
        raise Http404


@api_view(http_method_names=['POST'])
def search_value(request):
    pk = request.data['id']
    file = get_object_or_404(File, id=pk)
    df = dataframe_from_file(file.file)
    value = request.data['value']
    data = ''
    if value in df.columns:
        data = df[[value]]
    elif value in df.index:
        data = df.loc[[value]]
    else:
        data = df[df.isin([value]).any(1)]
    return Response(format_to_json(data))


@api_view(http_method_names=['POST'])
def describe(request):
    pk = request.data['id']
    file = get_object_or_404(File, id=pk)
    df = dataframe_from_file(file.file)
    json_object = format_to_json(df.describe().transpose())
    return Response(json_object)


@api_view(http_method_names=['POST'])
def transform(request):
    pk = request.data['id']
    column = request.data['column']
    convert_to = request.data['type']
    file = get_object_or_404(File, id=pk)
    df = dataframe_from_file(file.file)
    try:
        df = df.astype({column: convert_to}, errors='ignore')
    except Exception as e:
        pass
    return Response(format_to_json(df))


@api_view(http_method_names=['POST'])
def execute_query(request):
    pk = request.data['id']
    file = get_object_or_404(File, id=pk)
    query_string = request.data['query']
    df = dataframe_from_file(file.file)
    results = sqldf(query_string, locals())
    return Response(format_to_json(results))


@api_view(http_method_names=['POST'])
def filter_by_columns(request):
    pk = request.data['id']
    file = get_object_or_404(File, id=pk)
    columns = request.data['columns_names'].split(',')
    columns_list = []
    df = dataframe_from_file(file.file)
    if columns and columns[0]:
        for column in columns:
            columns_list.append(column)
        df = df[columns_list]
    return Response(format_to_json(df))


@api_view(['POST'])
def plot(request):
    pk = request.data['id']
    columns = request.data['columns'].split(",")
    x = request.data['x']
    kind = request.data['kind']
    y_colors = request.data['y_colors'].split(",")
    print(y_colors)
    file = get_object_or_404(File, id=pk)
    df = dataframe_from_file(file.file)
    try:
        graph_url = ''
        if kind not in ['treemap', 'correlogram', 'density-plot']:
            if kind == 'barh':
                df.plot(kind=kind, x=x, y=columns, color=y_colors, figsize=(12,9))
            elif kind == 'pie':
               df.plot(kind=kind, x=x, y=columns, colors=y_colors, figsize=(13, 14)) 
            elif kind == 'box':
                df.plot(kind=kind, x=x, y=columns, figsize=(12, 7)) 
            else:
                df.plot(kind=kind, x=x, y=columns, color=y_colors, figsize=(12,7))
            img = BytesIO()
            plt.savefig(img, format="png")
            img.seek(0)
            graph_url = base64.b64encode(img.getvalue()).decode()
            plt.clf()
        else:
            graph_url = plots(kind)
        return Response({
            'plot': f'data:image/png;base64,{graph_url}',
            'error': False
        })
    except Exception as e:
        return Response({
            'plot': '',
            'error': f'Error: {str(e)}'
        })


@api_view(['POST'])
def stats(request):
    pk = request.data['id']
    file = get_object_or_404(File, id=pk)
    df = dataframe_from_file(file.file)
    x = request.data['x']
    y = request.data['y']
    test = request.data['test']
    x_axis = df[[x]].to_numpy()
    y_axis = df[[y]].to_numpy()
    response = {}
    try:
        response['result'] = compute_stats(x_axis.ravel(), y_axis.ravel(), test)
        response['error'] = False
    except Exception as e:
        response['result'] = ''
        response['error'] = str(e)
    return Response(response)


@api_view(['POST'])
def reset(request):
    pk = request.data['id']
    file = get_object_or_404(File, id=pk)
    df = dataframe_from_file(file.file)
    return Response(format_to_json(df))


@api_view(['POST'])
def info(request):
    pk = request.data['id']
    file = get_object_or_404(File, id=pk)
    df = dataframe_from_file(file.file)
    ax = sns.heatmap(df.isnull(), annot=True, cbar=False)
    fig = ax.get_figure()
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    seaborn_plot = base64.b64encode(img.getvalue()).decode()
    plt.clf()
    ax = msno.matrix(df)
    fig = ax.get_figure()
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    matrix_plot = base64.b64encode(img.getvalue()).decode()
    plt.clf()
    ax = msno.heatmap(df)
    fig = ax.get_figure()
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    heatmap_plot = base64.b64encode(img.getvalue()).decode()
    plt.clf()
    ax = msno.bar(df)
    fig = ax.get_figure()
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    bar_plot = base64.b64encode(img.getvalue()).decode()
    plt.clf()
    return Response({
        'seaborn_plot': f'data:image/png;base64,{seaborn_plot}',
        'heatmap_plot': f'data:image/png;base64,{heatmap_plot}',
        'bar_plot': f'data:image/png;base64,{bar_plot}',
        'matrix_plot': f'data:image/png;base64,{matrix_plot}',
    })


@api_view(['POST'])
def fisher_test(request):
    table = request.data['table'].split(",")
    x_axis = [Decimal(x) for x in table[:len(table)//2]]
    y_axis = [Decimal(y) for y in table[len(table)//2:]]
    response = {}
    try:
        response['result'] = compute_stats(x_axis, y_axis, 'fisher_exact')
        response['error'] = False
    except Exception as e:
        response['result'] = ''
        response['error'] = str(e)
    return Response(response)


@api_view(['POST'])
def math_functions(request):
    pk = request.data['id']
    file = get_object_or_404(File, id=pk)
    df = dataframe_from_file(file.file)
    function_name = request.data['function']
    x = request.data['x']
    response = {'columns': [], 'rows': []}
    if function_name != '' and x != '':
        try: 
            column = df[[x]].to_numpy().ravel()
            np_array = call_math_function(function_name, column)
            response = format_np_array(np_array.ravel(), function_name, column, x)
            response['error'] = False
        except Exception as e:
            response["error"] = str(e)
    return Response(response)


@api_view(['POST'])
def split_data_set(request):
    pk = request.data['id']
    file = get_object_or_404(File, id=pk)
    df = dataframe_from_file(file.file)
    response = {'training_set': {}, 'test_set': {}, 'error': False}
    try:
        training_set, test_set = train_test_split(df, test_size=0.2)
        response = {
            'training_set': format_to_json(pd.DataFrame(training_set)),
            'test_set': format_to_json(pd.DataFrame(test_set)),
            'error': False
        }
    except Exception as e:
        response['error'] = str(e)
    return Response(response)


@api_view(['POST'])
def preprocessing(request):
    pk = request.data['id']
    file = get_object_or_404(File, id=pk)
    df = dataframe_from_file(file.file).select_dtypes(include=['number'])
    function = request.data['normalizer']
    response = {'normalized_training_set': {}, 'error': False}
    try:
        training_set, test_set = train_test_split(df, test_size=0.2)
        response = {
            'normalized_training_set': normalize_set(training_set, function),
            'error': False
        }
    except Exception as e:
        response['error'] = str(e)
    return Response(response)


@api_view(['POST'])
def fit(request):
    pk = request.data['id']
    file = get_object_or_404(File, id=pk)
    df = dataframe_from_file(file.file).select_dtypes(include=['number'])
    y = request.data['y']
    column = df[[y]]
    response = {'fit_result': {}, 'error': False}
    try:
        X_train, X_test, Y_train, Y_test = train_test_split(df, column, test_size=0.2)
        response = {
            'fit_result': LinearRegression().fit(X_train, Y_train),
            'error': False
        }
    except Exception as e:
        response['error'] = str(e)
    print(response)
    return Response(response)


@api_view(['POST'])
def predict(request):
    pk = request.data['id']
    file = get_object_or_404(File, id=pk)
    df = dataframe_from_file(file.file)
    y = request.data['y']
    x = request.data['x']
    algorithm = request.data['algorithm']
    response = {'predict_result': [], 'error': False}
    if algorithm == 'linear-regression':
        response = linear_regression(df, x, y)
    elif algorithm == 'multiple-linear-regression':
        response = multi_linear_regression(df)
    elif algorithm == 'logistic-regression':
        response = logistic_regression(df)
    elif algorithm == 'svr':
        response = svr(df)
    elif algorithm == 'decision-tree-regressor':
        response = decision_tree_regressor(df)
    elif algorithm == 'random-forest-regression':
        response = random_forest_regression(df)
    elif algorithm == 'k-nearest-neighbors':
        response = k_nearest_neighbors(df)
    elif algorithm == 'svc':
        response = support_vector_machine(df, 'linear')
    elif algorithm == 'k-svc':
        response = support_vector_machine(df, 'rbf')
    elif algorithm == 'decision-tree-classification':
        response = classification(df, 'decision-tree', 'Decision Tree Classification')
    elif algorithm == 'naives-bayes':
        response = classification(df, 'naives-bayes', 'Naive Bayes')
    elif algorithm == 'random-forest-classification':
        response = classification(df, 'random-forest', 'Random Forest Classification')
    elif algorithm == 'k-means-cluster':
        response = k_means_cluster(df)
    elif algorithm == 'hierarchical-cluster':
        response = hierarchical_cluster(df)
    elif algorithm == 'lda':
        response = lda(df)
    elif algorithm == 'pca':
        response = pca(df)
    elif algorithm == 'kpca':
        response = kpca(df)
    elif algorithm == 'fp-growth':
        response = my_func(df, 'fp-growth')
    elif algorithm == 'apriori':
        response = my_func(df, 'apriori')
    elif algorithm == 'thompson-sampling':
        response = thompson_sampling(df)
    elif algorithm == 'upper-confidence-bound':
        response = upper_confidence_bound(df)
    elif algorithm == 'artificial-neural-network':
        response = artificial_neural_network(df)
    elif algorithm == 'convolutional-neural-network':
        response = convolutional_neural_network(df)
    elif algorithm == 'recurrent-neural-network':
        response = recurrent_neural_network(df)
    elif algorithm == 'sentimental-analysis':
        response = sentimental_analysis(df)
    return Response(response)



@api_view(['POST'])
@parser_classes([MultiPartParser])
def upload_plots_files(request):
    file = PlotFile(file=request.data['file'])
    file.save()
    files = PlotFile.objects.all()
    files_details = []
    for file in files:
        files_details.append({
            'id': file.id,
            'url': file.file.url
        })
    return Response({
        'files': files_details
    })


@api_view(['GET'])
def plot_files(request):
    files = PlotFile.objects.all()
    files_details = []
    for file in files:
        files_details.append({
            'id': file.id,
            'url': file.file.url
        })
    return Response({
        'files': files_details
    })


@api_view(['DELETE'])
def delete_plot_file(request, pk):
    file = PlotFile.objects.get(id=pk)
    file.delete()
    return Response({
        'id': pk,
        'message': 'success'
    })


@api_view(['DELETE'])
def delete_plot_files(request):
    files = PlotFile.objects.all()
    for file in files:
        file.delete()
    return Response({
        'message': 'success'
    })



