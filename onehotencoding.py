#One hot encoding
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score


def get_mae(X, y):
    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(50), 
                                X, y, 
                                scoring = 'neg_mean_absolute_error').mean()


train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
target = train_data.SalePrice
#Eliminamos el objectivo, y el id porque no sirve de nada
prediction_data = train_data.drop(['Id'],axis=1)
#Dividimos la data, en train y test
X_train, X_test, y_train,y_test = train_test_split(prediction_data, target)
#Eliminamos columnas que esten vacias
my_imputer = Imputer()
#Hacemos el get_dummies de pandas
train_one_hot_encoding = pd.get_dummies(X_train)
test_one_hot_encoding = pd.get_dummies(X_test)
#Alineamos test_data y train_data
final_train_data, final_test_data = train_one_hot_encoding.align(test_one_hot_encoding, join='left',axis=1)
#Creamos la imputacion de la data
imputed_X_train = my_imputer.fit_transform(final_train_data)
imputed_X_test = my_imputer.transform(final_test_data)
#Cogemos el valor que queremos predecir
target =final_train_data.SalePrice
#Creamos el modelo
modelo= RandomForestRegressor()
modelo.fit(imputed_X_train,target)
#Y predecimos
prediccion = modelo.predict(imputed_X_test)
print("Primeras predicciones")
#Esto devuelve un array asi que lo pasamos a DataFrame de pandas
predicciones = pd.DataFrame(prediccion)
print(predicciones.head())

#Y sacamos el error absoluto
error_absuluto_one_hot = get_mae(imputed_X_train,target)
print('Mean Abslute Error with One-Hot Encoding: ' + str(int(error_absuluto_one_hot)))

    
#Creamos one hot encoding para las columnas categoricas
#one_hot_encoding_train = pd.get_dummies(train_reduced_data)
#one_hot_encoding_test = pd.get_dummies(test_reduced_data)
#Alineamos test_data y train_data
#final_train_data, final_test_data = one_hot_encoding_train.align(one_hot_encoding_test, join='left',axis=1)
#print("-----Data alineana y con one hot encoding-------")
#print(final_train_data.describe())
#final_target =final_train_data.SalePrice
#modelo = RandomForestRegressor()
#modelo.fit(final_train_data,final_target)
#prediccion = modelo.predict(final_test_data)
#print("La prediccion con one_hot_encoding es de " + str(int(prediccion)))

#error_absoluto_onehot = get_mae(final_train_data, final_target)
#print('Mean Abslute Error with One-Hot Encoding: ' + str(int(error_absoluto_onehot)))
