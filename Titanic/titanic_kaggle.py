import pandas as pd
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
#print(train_data.info())

columnas_usar = ['PassengerId','Age','Pclass','Fare','Sex']
target = ['Survived']

X = train_data[columnas_usar]
y=train_data[target]
X['Age'] = X['Age'].fillna(X['Age'].median())


sexos = {'male':0,'female':1}
X['Sex']=X['Sex'].apply(lambda x:sexos[x])
X.sample(10)
y_target = test_data[columnas_usar]
y_target['Age'] = y_target['Age'].fillna(y_target['Age'].median())
y_target['Fare'] = y_target['Fare'].fillna(y_target['Fare'].median())
y_target['Sex']=y_target['Sex'].apply(lambda x:sexos[x])

print("vacio" , y_target['Sex'].isnull().sum())

print(y_target.columns.values)
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.9, test_size=0.1, random_state =0)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

models = []
models.append(("Logistic Regresion", LogisticRegression))
models.append(("Tree classifier", DecisionTreeClassifier))
models.append(("KNeighborsClassifier", KNeighborsClassifier))
models.append(("LinearDiscriminantAnalysis", LinearDiscriminantAnalysis))
models.append(("GaussianNB", GaussianNB))
models.append(("SVC", SVC))
models.append(("RandomForestClassifier", RandomForestClassifier))
models.append(("XGBClassifier", XGBClassifier))

from sklearn.metrics import accuracy_score

for model in models:
    name, model = model
    m = model()
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    score = accuracy_score(y_test, pred)
    print(name, score)
    
modelo = XGBClassifier(n_estimators=1000, learning_rate=0.05)
modelo.fit(X_train, y_train, early_stopping_rounds=50, eval_set=[(X_test, y_test)],verbose=False)
pre= modelo.predict(X_test)
print("Tuned XGboost accuracy  " , accuracy_score(y_test, pre))

df_test = pd.read_csv("../input/test.csv")
df_test['Age'].fillna((train_data['Age'].mean()), inplace=True)
df_test['Fare'].fillna((train_data['Fare'].mean()), inplace=True)
df_test['Sex']=df_test['Sex'].apply(lambda x:sexos[x])

test = df_test[columnas_usar]
Y_pred = modelo.predict(test)
print(Y_pred[0:20])

submission = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],
    "Survived": Y_pred
})

print(submission.head())
print(submission.shape)
submission.to_csv('new_submision.csv', index=False)
