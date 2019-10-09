#%% importing important libraries
import numpy as np
import pandas as panda
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

#%% Loading the dataset
dataset = panda.read_csv('Mall_Customers.csv') #here we read the data from csv file using function from pandas library
# There is no empty cells in dataset, that's why there is no need to use fillna or imputer to fill the gaps
Input = dataset.iloc[ :, :4].values # get input values from dataset(independent)
output = dataset.iloc[:, -1].values # get outputs from dataset(dependent)
#print(output)

#%% Encoding necessary features, in my case gender
# Using label encoder transform column to numeric array
label = LabelEncoder()
Input[:, 1] = label.fit_transform(Input[:, 1])
# Using one hot encoder divide labelled column into two columns
one_hot = OneHotEncoder(categorical_features=[1])
Input = one_hot.fit_transform(Input).toarray()

# There is no dummy variable trap

#%% Split dataset into training and test sets
# 30% of rows to test, ant 70% to train
Input_train, Input_test, output_train, output_test = train_test_split(Input, output, test_size=0.3)

#%% Scaling input features
#Here I decided to scale input data matrix to the [0, 1] range, from sklearn documentation
min_max_scaler = preprocessing.MinMaxScaler()
Input_train_minmax = min_max_scaler.fit_transform(Input_train)
Input_test_minmax = min_max_scaler.transform(Input_test)
print(Input_test_minmax)