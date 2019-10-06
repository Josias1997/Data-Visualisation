from django.urls import path
from . import views

urlpatterns = [
	path('upload/', views.upload, name='upload'),
	path('filter/', views.filter, name='filter'),
	path('search/', views.search_value, name="search"),
	path('login/', views.login, name='login')
]