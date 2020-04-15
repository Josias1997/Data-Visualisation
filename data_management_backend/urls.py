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
	path('signup/', views.sign_up, name='signup'),
	path('plot/', views.plot, name='plot'),
	path('stats/', views.stats, name='stats'),
	path('fisher-test/', views.fisher_test, name="fisher_exact"),
	path('math-functions/', views.math_functions, name="maths-functions"),
	path('split-data-set/', views.split_data_set, name="split-data-set"),
	path('preprocessing/', views.preprocessing, name="preprocessing"),
	path('fit-data-set/', views.fit, name='fit-data-set'),
	path('predict-data-set/', views.predict, name='predict-data-set'),
	path('reset/', views.reset),
	path('info/', views.info),
	path('plot-files/', views.plot_files),
	path('upload-plot-files/', views.upload_plots_files),
	path('delete-plot-file/<int:pk>', views.delete_plot_file),
	path('delete-files/', views.delete_plot_files)
]