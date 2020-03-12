from django.urls import path
from .views import index

urlpatterns = [
	path('', index, name='index'),
    path('r-statistics/', index, name='index'),
    path('modelisation/', index, name='index'),
    path('machine-learning/', index, name='index'),
    path('deep-learning/', index, name='index'),
    path('text-mining/', index, name='index'),
    path('visualisation/', index, name='index'),
]