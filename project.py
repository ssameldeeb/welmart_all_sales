import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

data = pd.read_csv("WMT.csv")

print(data.shape)
print(data.columns.values)
print(data.dtypes)
print(data.isnull().sum())
print(data.head())

data["Date"] = pd.to_datetime(data["Date"])
data["year"] = data["Date"].dt.year
data["month"] = data["Date"].dt.month
data["day"] = data["Date"].dt.day
data = data.drop("Date", axis=1)
print(data.dtypes)

sns.heatmap(data.corr(),annot=True,cmap="magma")
plt.show()

x = data.drop("Volume", axis=1)
y = data["Volume"]

ss = StandardScaler()
x = ss.fit_transform(x)
print(x[:5])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle =True)
print(X_train.shape)

Li = LinearRegression()
Li.fit(X_train, y_train)

print("_"*100)
print(Li.score(X_train, y_train))
print(Li.score(X_test, y_test))
print("_"*100)


Lo = Lasso(alpha=0.0001,random_state=33,normalize=False)
Lo.fit(X_train, y_train)

print("_"*100)
print(Lo.score(X_train, y_train))
print(Lo.score(X_test, y_test))
print("_"*100)


# print("_"*150)
# for x in range(2,20):
#     Dt = DecisionTreeRegressor(max_depth=x,random_state=33)
#     Dt.fit(X_train, y_train)

#     print("x = ", x)
#     print(Dt.score(X_train, y_train))
#     print(Dt.score(X_test, y_test))
#     print("_"*100)
    
    


Dt = DecisionTreeRegressor(max_depth=7,random_state=33)
Dt.fit(X_train, y_train)

print(Dt.score(X_train, y_train))
print(Dt.score(X_test, y_test))
print("_"*100)
y_pred = Dt.predict(X_test)

# The autput result
result = pd.DataFrame({"y_test":y_test, "y_pred":y_pred})
# result.to_csv("The autput.csv",index=False)

    
    
# print("_"*150)
# for x in range(2,20):
#     ML = MLPRegressor(learning_rate='constant',early_stopping= False,alpha=0.01 ,hidden_layer_sizes=(100, x))

#     ML.fit(X_train, y_train)

#     print("x = ", x)
#     print(ML.score(X_train, y_train))
#     print(ML.score(X_test, y_test))
#     print("_"*100)