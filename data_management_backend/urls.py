from django.urls import path
from . import views

urlpatterns = [
	path('upload/', views.upload, name='upload'),
	path('filter/', views.filter, name='filter'),
	path('search/', views.search_value, name="search"),
	path('describe/', views.describe, name="describe"),
	path('transform/', views.transform, name="transform"),
	path('execute-query/', views.execute_query, name="query"),
	path('login/', views.login, name='login')
]