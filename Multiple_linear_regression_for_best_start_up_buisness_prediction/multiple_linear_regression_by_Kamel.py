#Backward Elimination and  Multiple linear regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import dataset

dataset = pd.read_csv('50_Startups.csv') #depend of your file name
X=dataset.iloc[: ,:-1].values # independet variables
y = dataset.iloc[: , 4].values #dependet variables


# encoding category data becaue the machine learning understand number for
#example like in this data we have Cities you need to make encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])  # we choose in this array 3 because our index is 3 the city 
#to make Dummy Encoding
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],   # The column numbers to be transformed (here is [3] because we have 3 cities
    remainder='passthrough')                                     # Leave the rest of the columns untouched
X = ct.fit_transform(X)

#To avoid Dummy trap and Multicollinearity (n-1) rule
X=X[: , 1:]   # we will take all rows and we start from column index 1 not 0 with this small line of code we avoid Multicollinearity

#splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y ,test_size=0.2 , random_state= 0)

#From this step we will start our model for Multiple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#now we want to predict after we train our model in last step from the X_test
y_pred = regressor.predict(X_test) #now we can see the predict value

# Now we Build Backward Elimination model
import statsmodels.regression.linear_model as lm
X = np.append(arr = np.ones((50,1)).astype(int), values = X , axis=1) # we do this step because the statesmodel dont consider the constant (y=bo+b1*x1) so we create Xo 
X_optimal = X[ :,[0,1,2,3,4,5]] # In this step we add all independet variables have p_value bigger than 0.05
X_optimal = np.array(X_optimal, dtype=float) #then we change it again to array
regressor_ols = lm.OLS(endog = y, exog = X_optimal).fit() #In this line we make linear regression with ordinary least square 
print(regressor_ols.summary())  # we print to see the p_vale and start our Backward Elimination

# we print to see the p_vale and start our Backward Elimination (p_vale<0.05) elminate any independet variables have p_value bigger than 0.05
X_optimal = X[ :,[0,1,3,4,5]]
X_optimal = np.array(X_optimal, dtype=float) 
regressor_ols = lm.OLS(endog = y, exog = X_optimal).fit()
print(regressor_ols.summary())

# p_vale and second iteration of our Backward Elimination (p_vale<0.05)
X_optimal = X[ :,[0,3,4,5]]
X_optimal = np.array(X_optimal, dtype=float) 
regressor_ols = lm.OLS(endog = y, exog = X_optimal).fit()
print(regressor_ols.summary())

# p_vale and third iteration of our Backward Elimination (p_vale<0.05)
X_optimal = X[ :,[0,3,5]]
X_optimal = np.array(X_optimal, dtype=float) 
regressor_ols = lm.OLS(endog = y, exog = X_optimal).fit()
print(regressor_ols.summary())

# p_vale and fourth iteration of our Backward Elimination (p_vale<0.05)
X_optimal = X[ :,[0,3]]
X_optimal = np.array(X_optimal, dtype=float) 
regressor_ols = lm.OLS(endog = y, exog = X_optimal).fit()
print(regressor_ols.summary())
