# SVR


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset

dataset = pd.read_csv('Position_Salaries.csv') #depend of your file name
X=dataset.iloc[: ,1:2].values # independet variables we make a little trick here to change to matrix 
y = dataset.iloc[: , 2].values #dependet variables
y=y.reshape(len(y),1)



#Feature scaling in SVR becasue SVR dont have feautre scaling
#Feature scaling we do it to make same range of all data from -1 to 1

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
print(X)
print(y)



# make SVR model

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)


# predict from SVR
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1)) #we need to inverse the transform to get the real value
#also we need 2d array that we make [[6.5]]

#Visualising the SVR model
plt.scatter(X, y, color = 'red')
plt.plot( X, regressor.predict(X) , color = 'blue')
plt.title('Truth or Bluff( SVR Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()







