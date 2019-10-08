from django.shortcuts import render, get_object_or_404, HttpResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from helpers.utils import dataframe_from_file, style_dataframe
from .models import File, Token
from django.contrib.auth.models import User
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.http import Http404
from pandasql import sqldf
import os


# Create your views here.

@api_view(http_method_names=['POST'])
@parser_classes([MultiPartParser])
def upload(request):
    file = File(file=request.data['file'])
    name, ext = os.path.splitext(file.file.name)
    file.title = name
    file.save()
    df = dataframe_from_file(file.file)
    html_data = '<div>Extension invalide</div>'
    json_response = {
        'data_table': html_data
    }
    if len(df) != 0:
        html_data = df.to_html()
        columns = df.columns.values.tolist()
        json_response = {
            'data_table': html_data,
            'columns_name': columns,
            'id': file.id,
            'size': df.size,
            'rows': df.shape[0],
            'columns': df.shape[1],
        }
    else:
        file.delete()
    return Response(json_response)


@api_view(http_method_names=['POST'])
def filter(request):
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
    return HttpResponse(filtered_data.to_html())


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
def search_value(request):
    pk = request.data['id']
    file = get_object_or_404(File, id=pk)
    df = dataframe_from_file(file.file)
    value = request.data['value']
    html_data = ''
    if value in df.columns:
        html_data = df[[value]]
    elif value in df.index:
        html_data = df.loc[[value]]
    else:
        html_data = df[df.isin([value]).any(1)]
    return HttpResponse(html_data.to_html())


@api_view(http_method_names=['POST'])
def describe(request):
    pk = request.data['id']
    file = get_object_or_404(File, id=pk)
    df = dataframe_from_file(file.file)
    return HttpResponse(df.describe().to_html())


@api_view(http_method_names=['POST'])
def transform(request):
    pk = request.data['id']
    column = request.data['column']
    convert_to = request.data['type']
    file = get_object_or_404(File, id=pk)
    df = dataframe_from_file(file.file)
    df = df.astype({column: convert_to}, errors='ignore')
    return HttpResponse(df.to_html())


@api_view(http_method_names=['POST'])
def execute_query(request):
    pk = request.data['id']
    file = get_object_or_404(File, id=pk)
    query_string = request.data['query']
    df = dataframe_from_file(file.file)
    results = sqldf(query_string, locals())
    return HttpResponse(results.to_html())


@api_view(http_method_names=['POST'])
def filter_columns(request):
    pk = request.data['id']
    file = get_object_or_404(File, id=pk)
    columns = request.data['columns_names'].split(" ")
    df = dataframe_from_file(file.file)
    return HttpResponse(df[columns].to_html())
