import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #for plot

# import dataset

dataset = pd.read_csv('Salary_Data.csv') #depend of your file name
X=dataset.iloc[: ,:-1].values # independet variables
y = dataset.iloc[: , 1].values #dependet variables he will take first coulmn with index 1


#splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y ,test_size=1/3 , random_state= 0)

#Feature scaling sample linear regression dont need Feature scaling
#Feature scaling we do it to make same range of all data from -1 to 1
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_trian = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""
# Training the Simple Linear Regression model on the Training set
#In this step we train our model with train values
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) #ordinary least square method this get the best line fitting 
                                #by taking the training value and actual value then subtracted
                                #from each other and sum them together then square it
# Predicting the Test set results
#In this step now we want to predict y_test from X_test values
y_pred = regressor.predict(X_test)
#In this step now we want to predict y_train from X_train values
y_pred_train = regressor.predict(X_train)
# Visualising the Training set results
plt.scatter(X_test, y_test, color ='red')
plt.plot(X_train, y_pred_train, color = 'blue')   
plt.title('salary vs Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('salary')
# Visualising the Test set results
# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

