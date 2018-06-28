import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


main_file_path = '../input/house-prices-advanced-regression-techniques/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)

primer_modelo = DecisionTreeRegressor()
# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
print('Some output from running this cell')
# Diviendo la data
y = data.SalePrice
targets= ['LotArea','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = data[targets]

train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=0)

targets= ['LotArea','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
print("Vamos a predecir un poco")
primer_modelo.fit(train_X,train_y)
prediccion = primer_modelo.predict(val_X)

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

for max_leaf_nodes in [10,20,30,40,50,60,70,80,90,100]:
    error_absoluto = get_mae(max_leaf_nodes,train_X,val_X,train_y,val_y)
    print("Numero de hojas: %d \t\t Error absuluto : %d" %(max_leaf_nodes,error_absoluto))
