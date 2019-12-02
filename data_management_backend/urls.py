from django.urls import path
from . import views

urlpatterns = [
	path('upload/', views.upload, name='upload'),
	path('filter/', views.filter_by_columns_and_rows, name='filter'),
	path('search/', views.search_value, name="search"),
	path('describe/', views.describe, name="describe"),
	path('transform/', views.transform, name="transform"),
	path('execute-query/', views.execute_query, name="query"),
	path('filter-columns/', views.filter_by_columns, name="filter-column"),
	path('login/', views.login, name='login'),
	path('plot/', views.plot, name='plot'),
	path('stats/', views.stats, name='stats')
]