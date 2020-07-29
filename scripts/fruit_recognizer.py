import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

root_directory = 'C:/Github/PYAIML'

# fruits_df = pd.read_table(root_directory +'/data/fruit_data_with_colors.txt')
# # print (len(fruits_df))
#
# # print (fruits_df.head())
#
# """ split the original data set for training set and testing set """
#
# X = fruits_df[['mass', 'width', 'height', 'color_score']]
# y = fruits_df['fruit_label']
#
# """ 75/25 partioning """
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# print (X_train, X_test, y_train, y_test)
""" scatter matrix plot """
# cmap = cm.get_cmap('gnuplot')
# scatter = pd.plotting.scatter_matrix(X_train, c=y_train, marker='o', s=40, hist_kwds={'bins':15}, figsize=(12,12), cmap=cmap)
# plt.show()


""" 3d scatter plots """
from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
# ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c=y_train, marker='o', s=100)
# ax.set_xlabel('width')
# ax.set_ylabel('height')
# ax.set_zlabel('color_score')
# plt.show()




""" knn classifier """
# lookup_fruit_name = dict(zip(fruits_df.fruit_label.unique(), fruits_df.fruit_name.unique()))
# # print (lookup_fruit_name)
#
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)
#
# print (knn.score(X_test, y_test))
#
#
# fruit_prediction = knn.predict([[20, 4.3, 5.5, 0.5]])
# print (lookup_fruit_name[fruit_prediction[0]])
#
# fruit_prediction = knn.predict([[100, 6.3, 8.5, 0.5]])
# print (lookup_fruit_name[fruit_prediction[0]])


""" generalization, overfitting, underfitting
    overfitting: model too complex
    underfitting: model too simple
    knn model:
    k too large: underfitting
    k too small: overfitting
"""


""" linear regression and linear model
    a weighted average model
    Least-squares models: Ordinary least-squares OLS

"""

# from sklearn.datasets import make_regression
# plt.figure()
# plt.title('sample regression problem with one input variable')
# X_R1, y_R1 = make_regression(n_samples=100, n_features=1, n_informative=1, bias=150, noise=30, random_state=0)
# plt.scatter(X_R1, y_R1, marker='o', s=50)
# # plt.show()
#
# from sklearn.linear_model import LinearRegression
# X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state=0)
# linreg = LinearRegression().fit(X_train, y_train)
# print ("linear model intercept (b): {}".format(linreg.intercept_))
# print ("linear model coeff (w): {}".format(linreg.coef_))
# print ("R-squared score (training): {:.3f}".format(linreg.score(X_train, y_train)))
# print ("R-squared score (test): {:.3f}".format(linreg.score(X_test, y_test)))
#
# plt.figure(figsize=(5,4))
# plt.scatter(X_R1, y_R1, marker='o', s=50, alpha=0.8)
# plt.plot(X_R1, linreg.coef_ * X_R1 + linreg.intercept_, 'r-')
# plt.title('Least-squares linear regression')
# plt.xlabel('Feature value (x)')
# plt.ylabel('Target value (y)')
# plt.show()



"""
    Ridge Regression
    Regularisation: adds a penalty for large variations in w parameters
    feature normalization
    minmaxscaler
"""
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler


crime = pd.read_table(root_directory+'/data/CommViolPredUnnormalizedData.txt', sep=',', na_values='?')

print (crime.head())


# remove features with poor coverage or lower relevance, and keep ViolentCrimesPerPop target column
columns_to_keep = [5, 6] + list(range(11,26)) + list(range(32, 103)) + [145]
crime = crime.ix[:,columns_to_keep].dropna()

X_crime = crime.ix[:,range(0,88)]
y_crime = crime['ViolentCrimesPerPop']





scaler = MinMaxScaler()


















# #






# #
