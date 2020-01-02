import pandas as pd
import numpy as np
import mlxtend
from mlxtend.frequent_patterns import fpgrowth, association_rules, apriori
import matplotlib.pyplot as plt
from .utils import generate_graph_img
import random
import math
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from ann_visualizer.visualize import ann_viz
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

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


def artificial_neural_network(df):
	X = df.iloc[:, 3:13].values
	y = df.iloc[:, 13].values

	response = {'error': False}
	try:
		# Encoding categorical data

		labelencoder_X_1 = LabelEncoder()
		X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
		labelencoder_X_2 = LabelEncoder()
		X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
		onehotencoder = OneHotEncoder(categories='auto')
		X = onehotencoder.fit_transform(X).toarray()
		X = X[:, 1:]

		# Splitting the dataset into the Training set and Test set
		X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

		# Feature Scaling

		sc = StandardScaler()
		X_train = sc.fit_transform(X_train)
		X_test = sc.transform(X_test)

		# Part 2 - Now let's make the ANN!

		# Importing the Keras libraries and packages

		# Initialising and creating the ANN model
		classifier = Sequential()

		# Adding the input layer and the first hidden layer
		classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))

		# Adding the second hidden layer
		classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

		# Adding the output layer
		classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

		# Compiling the ANN
		classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

		# Fitting the ANN to the Training set
		classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

		# Part 3 - Making the predictions and evaluating the model

		# Predicting the Test set results
		y_pred = classifier.predict(X_test)
		y_pred = (y_pred > 0.5)

		# Making the Confusion Matrix
		

		cm = confusion_matrix(y_test, y_pred)
		print(cm)
		plt.matshow(cm)
		plt.title('Confusion matrix')
		plt.colorbar()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		matrix_plot = generate_graph_img(plt)

		f1_score(y_pred, y_test)
		precision_score(y_pred, y_test)
		recall_score(y_pred, y_test)

		# Visualizing the Artificial Neural Network
		# pip install ann_visualizer
		# conda install graphviz
		ann_viz(classifier, filename="data_management_frontend/static/deep_learning/reseau.gv", title = "Reseau de neurone")

		response = {
			'matrix_plot': f'data:image/png;base64,{matrix_plot}',
			'confusion_matrix': cm,
			'error': False,
		}
		
	except Exception as e:
		response = {
			'error': str(e)
		}
	return response
