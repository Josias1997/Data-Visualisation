from django.shortcuts import render, get_object_or_404, HttpResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from helpers.utils import dataframe_from_file, style_dataframe
from .models import File, Token
from django.contrib.auth.models import User
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.http import Http404
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
		json_response = {
			'data_table': html_data,
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
	beginLine = int(request.data['beginLine']) if request.data['beginLine'] else ''
	endLine = int(request.data['endLine']) if request.data['endLine'] else ''
	beginColumn = int(request.data['beginColumn']) if request.data['beginColumn'] else ''
	endColumn = int(request.data['endColumn']) if request.data['endColumn'] else ''
	lastLine = df.shape[0] + 1
	lastColumn = df.shape[1] + 1
	filtered_datas = df
	if (beginLine != '' and (beginLine < 0 or beginLine >= lastLine - 1) ) or (endLine != '' and endLine > lastLine) or (beginColumn != '' and (beginColumn < 0 or beginColumn >= lastColumn - 1)) or (endColumn != '' and endColumn > lastColumn):
    		return HttpResponse("Indices invalides (Index lignes ou colonnes en dehors de la plage autorisée)")
	if beginLine != '' and endLine != '' and beginColumn != '' and endColumn != '':
			filtered_datas = df.iloc[beginLine:endLine,beginColumn:endColumn]
	elif beginLine == '' and endLine != '' and beginColumn != '' and endColumn != '':
			filtered_datas = df.iloc[:endLine,beginColumn:endColumn]
	elif beginLine == '' and endLine == '' and beginColumn != '' and endColumn != '':
			filtered_datas = df.iloc[:,beginColumn:endColumn]
	elif beginLine == '' and endLine == '' and beginColumn == '' and endColumn != '':
			filtered_datas = df.iloc[:,:endColumn]
	elif beginLine != '' and endLine == '' and beginColumn == '' and endColumn == '':
			filtered_datas = df.iloc[beginLine:,:]
	elif beginLine != '' and endLine != '' and beginColumn == '' and endColumn == '':
			filtered_datas = df.iloc[beginLine:endLine,:]
	elif beginLine != '' and endLine != '' and beginColumn != '' and endColumn == '':
			filtered_datas = df.iloc[beginLine:endLine,beginColumn:]
	elif beginLine != '' and endLine == '' and beginColumn == '' and endColumn != '':
			filtered_datas = df.iloc[beginLine:,:endColumn]
	elif beginLine != '' and endLine != '' and beginColumn == '' and endColumn != '':
			filtered_datas = df.iloc[beginLine:endLine,:endColumn]
	elif beginLine != '' and endLine == '' and beginColumn != '' and endColumn != '':
			filtered_datas = df.iloc[beginLine:,beginColumn:endColumn]
	elif beginLine == '' and endLine != '' and beginColumn != '' and endColumn == '':
			filtered_datas = df.iloc[:endLine,beginColumn:]
	elif beginLine != '' and endLine == '' and beginColumn != '' and endColumn == '':
			filtered_datas = df.iloc[beginLine:,beginColumn:]
	elif beginLine == '' and endLine != '' and beginColumn == '' and endColumn != '':
			filtered_datas = df.iloc[:endLine,:endColumn]
	return HttpResponse(filtered_datas.to_html())



@api_view(http_method_names=['POST'])
def login(request):
	username = request.data['username']
	password = request.data['password']
	try:
		user = User.objects.get(username=username)
		if not user.check_password(password):
			raise ValidationError(('Invalid Password'), code='invalid')
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
