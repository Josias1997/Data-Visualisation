from django.urls import path
from .views import index

urlpatterns = [
	path('', index, name='index'),
	path('r-statistics', index, name='statictics'),
	path('modelisation', index, name='modelisation')
]