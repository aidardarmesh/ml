#%% importing important libraries
# I wrote the code by looking on the example on the site https://www.geeksforgeeks.org/python-implementation-of-polynomial-regression/
import pandas as panda
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as mp
#%% Loading the dataset
dataset = panda.read_csv('Position_Salaries.csv')
# here I read the data from csv file using function from pandas library
# There is no empty cells in dataset, that's why there is no need to use fillna or imputer to fill the gaps
Input = dataset.iloc[:, 1].values
# get input values from dataset(independent)
output = dataset.iloc[:, -1].values
# get outputs from dataset(dependent)
Input = Input.reshape(-1, 1)
output = output.reshape(-1, 1)
#  reshape our features

#%%  fit the model from Polynomial regression to input and output components
poly = PolynomialFeatures(degree=4)
Input_poly = poly.fit_transform(Input)
poly.fit(Input_poly, output)
lin2 = LinearRegression()
lin2.fit(Input_poly, output)
#%% predicting results using polynomial regression
pred = lin2.predict(poly.fit_transform(Input))
#%%  visualize the results
mp.scatter(Input, output, color='blue')
mp.plot(Input, lin2.predict(poly.fit_transform(Input)), color='red')
mp.title('Polynomial Regression')
mp.xlabel('Level')
mp.ylabel('Salary')
mp.show()
