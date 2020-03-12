import pandas as pd
import numpy as np
import mlxtend
from mlxtend.frequent_patterns import fpgrowth, association_rules, apriori
import matplotlib.pyplot as plt
from .utils import generate_graph_img
import random
import math
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional, Flatten, Conv2D, MaxPool2D
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import SGD, RMSprop, Adam
from ann_visualizer.visualize import ann_viz
import seaborn as sns
import warnings
import os
import itertools

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
		plot_model(classifier, to_file='data_management_frontend/static/deep_learning/model.png')

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


def convolutional_neural_network(df):
	warnings.filterwarnings('ignore')
	response = {
		'error': False,
	}
	try:
		# read train 
		train = pd.read_csv("helpers/datasets/train.csv")
		print(train.shape)
		train.head()

		# read test 
		test= pd.read_csv("helpers/datasets/test.csv")
		print(test.shape)
		test.head()

		# put labels into y_train variable
		Y_train = train["label"]
		# Drop 'label' column
		X_train = train.drop(labels = ["label"],axis = 1) 

		# visualize number of digits classes 
		plt.figure(figsize=(15,7))
		g = sns.countplot(Y_train)
		plt.title("Number of digit classes")
		Y_train.value_counts()

		# plot some samples
		img = X_train.iloc[0].as_matrix()
		img = img.reshape((28,28))
		plt.imshow(img,cmap='gray')
		plt.title(train.iloc[0,0])
		plt.axis("off")
		samples_1 = generate_graph_img(plt)

		# plot some samples
		img = X_train.iloc[3].as_matrix()
		img = img.reshape((28,28))
		plt.imshow(img,cmap='gray')
		plt.title(train.iloc[3,0])
		plt.axis("off")
		samples_2 = generate_graph_img(plt)

		# Normalize the data
		X_train = X_train / 255.0
		test = test / 255.0
		print("x_train shape: ",X_train.shape)
		print("test shape: ",test.shape)

		# Reshape
		X_train = X_train.values.reshape(-1,28,28,1)
		test = test.values.reshape(-1,28,28,1)
		print("x_train shape: ",X_train.shape)
		print("test shape: ",test.shape)

		# Label Encoding 

		Y_train = to_categorical(Y_train, num_classes = 10)

		# Split the train and the validation set for the fitting
		X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
		print("x_train shape",X_train.shape)
		print("x_test shape",X_val.shape)
		print("y_train shape",Y_train.shape)
		print("y_test shape",Y_val.shape)

		# Some examples
		plt.imshow(X_train[2][:,:,0],cmap='gray')
		examples = generate_graph_img(plt)


		model = Sequential()

		model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 
		                 activation ='relu', input_shape = (28,28,1)))
		model.add(MaxPool2D(pool_size=(2,2)))

		model.add(Dropout(0.25))

		model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 
		                 activation ='relu'))
		model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

		model.add(Dropout(0.25))

		model.add(Flatten())

		model.add(Dense(256, activation = "relu"))

		model.add(Dropout(0.5))

		model.add(Dense(10, activation = "softmax"))

		# Define the optimizer
		optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

		# Compile the model
		model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

		epochs = 10  # for better result increase the epochs
		batch_size = 250


		# data augmentation
		datagen = ImageDataGenerator(
		        featurewise_center=False,  # set input mean to 0 over the dataset
		        samplewise_center=False,  # set each sample mean to 0
		        featurewise_std_normalization=False,  # divide inputs by std of the dataset
		        samplewise_std_normalization=False,  # divide each input by its std
		        zca_whitening=False,  # dimension reduction
		        rotation_range=0.5,  # randomly rotate images in the range 5 degrees
		        zoom_range = 0.5, # Randomly zoom image 5%
		        width_shift_range=0.5,  # randomly shift images horizontally 5%
		        height_shift_range=0.5,  # randomly shift images vertically 5%
		        horizontal_flip=False,  # randomly flip images
		        vertical_flip=False)  # randomly flip images

		datagen.fit(X_train)

		# Fit the model
		history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
		                              epochs = epochs, validation_data = (X_val,Y_val),
		                              steps_per_epoch=X_train.shape[0] // batch_size)


		# Evaluate the model
		# Plot the loss and accuracy curves for training and validation 
		plt.plot(history.history['val_loss'], color='b', label="validation loss")
		plt.title("Test Loss")
		plt.xlabel("Number of Epochs")
		plt.ylabel("Loss")
		plt.legend()
		model_plot = generate_graph_img(plt)

		# Predict the values from the validation dataset
		Y_pred = model.predict(X_val)
		# Convert predictions classes to one hot vectors 
		Y_pred_classes = np.argmax(Y_pred,axis = 1) 
		# Convert validation observations to one hot vectors
		Y_true = np.argmax(Y_val,axis = 1) 
		# compute the confusion matrix
		matrix = confusion_matrix(Y_true, Y_pred_classes)

		plt.figure(figsize = (10,7))
		sns.heatmap(matrix, annot=True)

		# plot the confusion matrix
		f,ax = plt.subplots(figsize=(8, 8))
		sns.heatmap(matrix, annot=True, linewidths=0.01,cmap="Greens",linecolor="blue", fmt= '.1f',ax=ax)
		plt.xlabel("Predicted Label")
		plt.ylabel("True Label")
		plt.title("Confusion Matrix")
		matrix_plot = generate_graph_img(plt)
		response = {
			'samples_1': f'data:image/png;base64,{samples_1}',
			'samples_2': f'data:image/png;base64,{samples_2}',
			'examples': f'data:image/png;base64,{examples}',
			'model': f'data:image/png;base64,{model_plot}',
			'confusion_matrix': f'data:image/png;base64,{matrix_plot}'
		}
	except Exception as e:
		response = {
			'error': str(e)
		}
	return response


# Some functions to help out with
def plot_predictions(test,predicted):
    plt.plot(test, color='red',label='Real IBM Stock Price')
    plt.plot(predicted, color='blue',label='Predicted IBM Stock Price')
    plt.title('IBM Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('IBM Stock Price')
    plt.legend()
    return generate_graph_img(plt)


def return_rmse(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))



def recurrent_neural_network(df):
	plt.style.use('fivethirtyeight')

	response = {'error': False}
	# try:
	df.set_index('Date')
	# Checking for missing values
	training_set = df.loc[:'2016'].iloc[:,1:2].values
	test_set = df.loc['2017':].iloc[:,1:2].values
	# We have chosen 'High' attribute for prices. Let's see what it looks like
	df["High"].loc[:'2016'].plot(figsize=(16,4),legend=True)
	df["High"].loc['2017':].plot(figsize=(16,4),legend=True)
	plt.legend(['Training set (Avant 2017)','Test set (2017 et au delà)'])
	plt.title('IBM stock price')
	stock_price_plot = generate_graph_img(plt)

	# Scaling the training set
	sc = MinMaxScaler(feature_range=(0,1))
	training_set_scaled = sc.fit_transform(training_set)

	# Since LSTMs store long term memory state, we create a data structure with 60 timesteps and 1 output
	# So for each element of training set, we have 60 previous training set elements 
	X_train = []
	y_train = []
	print(training_set_scaled.shape)
	for i in range(60,2017):
	    X_train.append(training_set_scaled[i-60:i,0])
	    y_train.append(training_set_scaled[i,0])

	X_train, y_train = np.array(X_train), np.array(y_train)

	# Reshaping X_train for efficient modelling
	X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

	# The LSTM architecture
	regressor = Sequential()

	# First LSTM layer with Dropout regularisation
	regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
	print('OK')
	regressor.add(Dropout(0.2))
	# Second LSTM layer
	regressor.add(LSTM(units=50, return_sequences=True))
	regressor.add(Dropout(0.2))

	# Third LSTM layer
	regressor.add(LSTM(units=50, return_sequences=True))
	regressor.add(Dropout(0.2))

	# Fourth LSTM layer
	regressor.add(LSTM(units=50))
	regressor.add(Dropout(0.2))

	# The output layer
	regressor.add(Dense(units=1))

	# Compiling the RNN
	regressor.compile(optimizer='rmsprop',loss='mean_squared_error')

	# Fitting to the training set
	regressor.fit(X_train,y_train,epochs=2,batch_size=32)

	# Now to get the test set ready in a similar way as the training set.
	# The following has been done so forst 60 entires of test set have 60 previous values which is impossible to get unless we take the whole 
	# 'High' attribute data for processing
	dataset_total = pd.concat((df["High"].loc[:'2016'],df["High"].loc['2017':]),axis=0)
	inputs = dataset_total[len(dataset_total)-len(test_set) - 60:].values
	inputs = inputs.reshape(-1,1)
	inputs  = sc.transform(inputs)

	# Preparing X_test and predicting the prices
	X_test = []
	for i in range(60,311):
	    X_test.append(inputs[i-60:i,0])
	X_test = np.array(X_test)
	X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
	predicted_stock_price = regressor.predict(X_test)
	predicted_stock_price = sc.inverse_transform(predicted_stock_price)

	# Visualizing the results for LSTM
	print(test_set)
	lstm_plot = plot_predictions(test_set.ravel(),predicted_stock_price)


	# The GRU architecture
	regressorGRU = Sequential()

	# First GRU layer with Dropout regularisation
	regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
	regressorGRU.add(Dropout(0.2))
	# Second GRU layer
	regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
	regressorGRU.add(Dropout(0.2))
	# Third GRU layer
	regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
	regressorGRU.add(Dropout(0.2))
	# Fourth GRU layer
	regressorGRU.add(GRU(units=50, activation='tanh'))
	regressorGRU.add(Dropout(0.2))
	# The output layer
	regressorGRU.add(Dense(units=1))
	# Compiling the RNN
	regressorGRU.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='mean_squared_error')
	# Fitting to the training set
	regressorGRU.fit(X_train,y_train,epochs=2,batch_size=150)

	# Preparing X_test and predicting the prices
	X_test = []
	for i in range(60,311):
	    X_test.append(inputs[i-60:i,0])
	X_test = np.array(X_test)
	X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
	GRU_predicted_stock_price = regressorGRU.predict(X_test)
	GRU_predicted_stock_price = sc.inverse_transform(GRU_predicted_stock_price)

	# Visualizing the results for GRU
	gru_plot = plot_predictions(test_set.ravel(),GRU_predicted_stock_price)
	# Preparing sequence data
	print(X_train.shape)
	initial_sequence = X_train[1956,:]
	sequence = []
	for i in range(251):
	    new_prediction = regressorGRU.predict(initial_sequence.reshape(initial_sequence.shape[1],initial_sequence.shape[0],1))
	    initial_sequence = initial_sequence[1:]
	    initial_sequence = np.append(initial_sequence,new_prediction,axis=0)
	    sequence.append(new_prediction)
	sequence = sc.inverse_transform(np.array(sequence).reshape(251,1))

	# Visualizing the sequence
	sequence_plot = plot_predictions(test_set.ravel(),sequence)
	plt.clf()

	response = {
		'stock_price_plot': f'data:image/png;base64,{stock_price_plot}',
		'lstm_plot': f'data:image/png;base64,{lstm_plot}',
		'gru_plot': f'data:image/png;base64,{gru_plot}',
		'sequence_plot': f'data:image/png;base64,{sequence_plot}',
	}
	# except Exception as e:
	# 	response = {
	# 		'error': str(e)
	# 	}
	return response
