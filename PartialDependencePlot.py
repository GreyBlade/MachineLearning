#Partial dependence plots
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
print(data.info())
cols_to_use = ['LotArea', 'GarageArea', 'YearBuilt']
X = data[cols_to_use]
target = data.SalePrice

my_imputer = Imputer()

X_train_final = my_imputer.fit_transform(X)
print(X_train_final)

my_model = GradientBoostingRegressor()
my_model.fit(X_train_final,target)
my_plots = plot_partial_dependence(my_model, 
                                  features=[0,2], 
                                  X=X_train_final, 
                                  feature_names=cols_to_use, 
                                  grid_resolution=10)
