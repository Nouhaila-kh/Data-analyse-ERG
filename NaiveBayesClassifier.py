import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from NaiveBayes import NaiveBayes


data1= pd.read_csv("data.csv")
#data= np.array(data1)[0:82,:]
data=data1.iloc[0:82,:]
'''
#remplacement des nan dans le tableau par des 0
y= pd.isnull(data)
data[y]=0
'''

#preparation des données
x=np.array(data.iloc[:,:5])
#print(x.shape)
y=np.array(data.iloc[:,5])

y=np.where(y=="Sain",0,y)
y=np.where(y=="Cannabis",1,y)


x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)
model = NaiveBayes()
model.fit(x_train,y_train)

print('this is prediction for x_test')
predictions=model.predict(x_test)
print(predictions)
print(y_test)

print("Naive Bayes accuracy prediction is ",model.accuracy(y_test,predictions))



