from joblib.logger import PrintTime
from sklearn import datasets, linear_model
import numpy as np 
from sklearn import linear_model
# from sklearn.metrics import mean_squared_error

#to find if verginica
database = datasets.load_iris()
features = database.data[:,3:]
label =(database.target==2).astype(int)

model = linear_model.LogisticRegression()
model.fit(features,label)
predict = model.predict([[2.4]])
# if (predict==1):
#     print("verginica")
# else:
#     print("not verginica")
# # print(predict)
new_features = np.linspace(0,3,12000).reshape(-1,1)
prediction = model.predict_proba(new_features)
import matplotlib.pyplot as plt
plt.plot(new_features,prediction[:,-1],"g-",label = "virginica")
plt.show()