import pandas as pd
import numpy as np
import pydot
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn.utils import shuffle


def main():
	### DATA PREPARATION ###
	features = pd.read_csv("full_data_no_one_hot.csv")

	# Input Features: temperature, dew point, humidity, wind direction, wind speed, wind gust speed, pressure, precipitation
	features.head(5)
	features.describe()
	
	# Labels are the values we want to predict
	labels = np.array(features['time_taxi_out'])
	# Remove the labels from the features
	# axis 1 refers to the columns
	features = features.drop('time_taxi_out', axis = 1)
	# Saving feature names for later use
	feature_list = list(features.columns)
	# Convert to numpy array
	features = np.array(features)
	features, labels = shuffle(features, labels)
	features = features[0:400000, :]
	labels = labels[0:400000]

	# Split the data into training and testing sets
	train_features, test_features, train_labels, test_labels = \
		train_test_split(features, labels, test_size = 0.2, random_state = 42)

	print('Training Features Shape:', train_features.shape)
	print('Training Labels Shape:', train_labels.shape)
	print('Testing Features Shape:', test_features.shape)
	print('Testing Labels Shape:', test_labels.shape)

	### TRAIN ###
	# Instantiate model with 1000 decision trees
	rf = RandomForestRegressor(n_estimators = 500, random_state = 42)
	# Train the model on training data
	rf.fit(train_features, train_labels);

	### TEST ###
	# Use the forest's predict method on the train data
	predictions_train = rf.predict(train_features)
	errors_train = abs(predictions_train - train_labels)
	print('Mean Absolute Error Train:', round(np.mean(errors_train), 2), 'degrees.')

	# Use the forest's predict method on the test data
	predictions = rf.predict(test_features)
	# Calculate the absolute errors
	errors = abs(predictions - test_labels)
	# Print out the mean absolute error (mae)
	print('Mean Absolute Error Test:', round(np.mean(errors), 2), 'degrees.')

	### GRAPH ###
	# Limit depth of tree to 3 levels
	rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
	rf_small.fit(train_features, train_labels)
	# Extract the small tree
	tree_small = rf_small.estimators_[5]
	# Save the tree as a png image
	export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
	(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
	graph.write_png('small_tree.png')


if __name__ == '__main__':
	main()

### DEPRECATED LINEAR REGRESSION ###
#Xs = data.drop(['sales', 'Unnamed: 0'], axis=1)
	# X = data[['humidity', 'dew_point', 'wind_speed_mph', 'wind_gust_mph', 'pressure_in', 'precip_in']]
	# #X = data[['wind_speed_mph', 'wind_gust_mph']]
	# y = data['time_taxi_out'].values.reshape(-1, 1)
	# reg = LinearRegression()
	# reg.fit(X, y)
	# # print("The linear model is: Y = {:.5} + {:.5}*Temp + {:.5}*Dew Point + {:.5}*Humidity " \
	# # 	"+ :{:.5}*Wind Dir + {:.5}*Wind Speed + {:.5}*Wind Gust Speed + {:.5}*Pressure + {:.5}*Precip"). \
	# # 	format(reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1], reg.coef_[0][2], \
	# # 			reg.coef_[0][3], reg.coef_[0][4], reg.coef_[0][5], reg.coef_[0][6], \
	# # 			reg.coef_[0][7], reg.coef_[0][8])
	# print(reg.coef_)
	# print(reg.intercept_)
	# #print("The linear model is: Y = {:.5} + {:.5}*Wind speed + {:.5}*Wind Gust Speed").format(reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1])

	# predictions = reg.predict(X)
	# print(reg.score(X, y))

	# X = np.column_stack((data['humidity'], data['dew_point'], data['wind_speed_mph'], data['wind_gust_mph'], data['pressure_in'], data['precip_in']))
	# #X = np.column_stack((data['wind_speed_mph'], data['wind_gust_mph']))
	# X2 = sm.add_constant(X)
	# est = sm.OLS(y, X2)
	# est2 = est.fit()
	# print(est2.summary())
####################################
