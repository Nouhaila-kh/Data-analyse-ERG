import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

data1= pd.read_csv("data.csv")
data=data1.iloc[0:82,:]
#data= np.array(data1)[0:82,:]
#remplacement des nan dans le tableau par des 0
'''
y= pd.isnull(data)
data[y]=0
'''

#preparation des données
x=data.iloc[:,0:5]
#print(x.shape)
y=data.iloc[:,5]

'''
plt.scatter(x.iloc[:,1], y, c='red', marker='+')
plt.show()
'''

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)
model=LinearSVC(tol=1e-5, C=1.0, max_iter=100000)
model.fit(x_train,y_train)
print(" c'est le score du model ")
print(model.score(x_test,y_test))
print('this is prediction for x_test')
print(y_test)
print(model.predict(x_test))

print('matrice de confusion')
print(confusion_matrix(model.predict(x_test),y_test))
