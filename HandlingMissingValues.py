import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

# this is the path to the Iowa data that you will use
main_file_path = '../input/house-prices-advanced-regression-techniques/train.csv' 

data = pd.read_csv(main_file_path)

def score_dataset(X_train,X_test,y_train,y_test):
    forestModel = RandomForestRegressor()
    forestModel.fit(X_train,y_train)
    prediccion = forestModel.predict(X_test)
    error_absoluto = mean_absolute_error(y_test,prediccion)
    return (error_absoluto)

y = data.SalePrice
iowa_prediciones=data.drop(['SalePrice'],axis=1)
iowa_prediciones_numericas = iowa_prediciones.select_dtypes(exclude=['object'])

X_train, X_test, y_train, y_test = train_test_split(iowa_prediciones_numericas, 
                                                    y,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)

cols_with_missing = [col for col in X_train.columns 
                                 if X_train[col].isnull().any()]
redued_original_data = X_train.drop(cols_with_missing, axis=1)
reduced_test_data = X_test.drop(cols_with_missing, axis=1)

print("El error absoluto al eliminar las columnas con datos vacios es:")
print(score_dataset(redued_original_data, reduced_test_data,y_train, y_test))


my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("El error absoluto con imputacion es:")
print(score_dataset(imputed_X_train,imputed_X_test,y_train,y_test))


imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()
cols_with_missing = (col for col in X_train.columns 
                                 if X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()
    
my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("El error absoluto del ultimo metodo es: ")
print(score_dataset(imputed_X_train_plus,imputed_X_test_plus,y_train,y_test))
