# Decision Tree Regression


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset

dataset = pd.read_csv('Position_Salaries.csv') #depend of your file name
X=dataset.iloc[: ,1:2].values # independet variables we make a little trick here to change to matrix 
y = dataset.iloc[: , 2].values #dependet variables



# make model of Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
regress = DecisionTreeRegressor()
regress.fit(X,y)



# predict from Linear regressin model vs from pply regression
y_pred = regress.predict([[5.5]])
print(y_pred)


#Visualising the Decision Tree Regression model
plt.scatter(X, y, color = 'red')
plt.plot( X, regress.predict(X) , color = 'blue')
plt.title('Truth or Bluff( Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualising the Decision Tree Regression model with high resolution
X_grid = np.arange(min(X), max(X), 0.1)  # we make this step to make the step smaller then the curve will be smooth and to increase the resolution
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot( X_grid, regress.predict(X_grid) , color = 'blue')
plt.title('Truth or Bluff( Decision Tree Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()






