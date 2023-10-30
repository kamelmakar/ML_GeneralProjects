import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset

dataset = pd.read_csv('Position_Salaries.csv') #depend of your file name
X=dataset.iloc[: ,1:2].values # independet variables we make a little trick here to change to matrix 
y = dataset.iloc[: , 2].values #dependet variables

''' # we will not make splitting data becouse our data is small
#splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y ,test_size=0.2 , random_state= 0)
'''
# make linear Regression from the data set
from sklearn.linear_model import LinearRegression
lin_regress = LinearRegression()
lin_regress.fit(X, y)

# make the fitting for polynominal Regression from the data set
from sklearn.preprocessing import PolynomialFeatures
poly_regress = PolynomialFeatures(degree = 4) # we make our degree 4 because it will give better result
X_poly = poly_regress.fit_transform(X)
lin_regress_2 = LinearRegression()
lin_regress_2.fit(X_poly, y)

#Visualising the Linear regression model
plt.scatter(X, y, color = 'red')
plt.plot( X, lin_regress.predict(X) , color = 'blue')
plt.title('Truth or Bluff(simple Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualising the Polynominial regression model
X_grid = np.arange(min(X), max(X), 0.1)  # we make this step to make the step smaller then the curve will be smooth and to increase the resolution
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot( X_grid, lin_regress_2.predict(poly_regress.fit_transform(X_grid)) , color = 'blue')
plt.title('Truth or Bluff(Polynominal Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# predict from Linear regressin model vs from pply regression
lin_regress.predict([[5.5]])
lin_regress_2.predict(poly_regress.fit_transform([[5.5]]))





