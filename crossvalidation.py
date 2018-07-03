#CROSS VALIDATION
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor



data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'],axis=1)
X_onehot = pd.get_dummies(X)
my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())
scores = cross_val_score(my_pipeline, X_onehot, y, scoring='neg_mean_absolute_error')
print(scores)
print('Mean Absolute Error RandomForestRegressor %2f' %(-1 * scores.mean()))
my_pipeline = make_pipeline(Imputer(), XGBRegressor())
scores = cross_val_score(my_pipeline, X_onehot, y, scoring='neg_mean_absolute_error')
print(scores)
print('Mean Absolute Error XGBoost %2f' %(-1 * scores.mean()))
