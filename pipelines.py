#Pipelines
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer


#Leemos la data
data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['Id','SalePrice'],axis=1)
#La dividimos en test, y train
X_train, X_test, y_train,y_test = train_test_split(X, y)
#Hacemos el get_dummies para los valores que no son numericos
X_train_one_hot_encoding = pd.get_dummies(X_train)
X_test_one_hot_encoding = pd.get_dummies(X_test)
#Alieneamos los data set
final_train_data, final_test_data = X_train_one_hot_encoding.align(X_test_one_hot_encoding, join='left',axis=1)
#Creamos el pipeline
my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())
#Y hacemos la prediccion
my_pipeline.fit(final_train_data,y_train)
predictions = my_pipeline.predict(final_test_data)
print(predictions)
print("Error absoluto con RandomForestRegressor")
print(mean_absolute_error(predictions,y_test))
