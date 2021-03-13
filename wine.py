# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 21:03:19 2021

@author: bielj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("wine_dataset.csv")
df.isnull().sum()
df.dtypes.sort_values()
df.corr()
df.isnull().values.any()
df.describe(include = "all")
len(df.columns) # 13 columns

# Dividindo entre variaveis target e variaveis preditores
x = df.drop("style", axis = 1)
y = df["style"]

y = y.replace("red", 0)
y = y.replace("white", 1)

# Tranformando as variaveis categoricas em variaveis numerais
from sklearn.preprocessing import LabelEncoder
label_Encoder_Y = LabelEncoder()
y =  label_Encoder_Y.fit_transform(y) # Red = 0 // White = 1


# Modelo treinado
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state = 44)

# Escolhendo o modelo de machine-learning
from sklearn.ensemble import ExtraTreesClassifier
model_v1 = ExtraTreesClassifier()
model_v1.fit(x_train, y_train.ravel())


# Verificando a accuracy: 0.9974
from sklearn import metrics
predict_test = model_v1.predict(x_test)
print("Accurary: {0:.4f}".format(metrics.accuracy_score(y_test, predict_test))) 

prevision_model_test = []
prevision_model_train = []

def creat_graph(model):
    for number in range(100):
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=number)
        model = model.fit(x_train, y_train)
        
        prevision_model_train.append(model.score(x_train, y_train))
        prevision_model_test.append(model.score( x_test, y_test))
        
creat_graph(model_v1)
        

# Montando grafico com cada random_state
plt.figure(figsize = (15, 10))
plt.grid()
plt.plot(prevision_model_train, label = 'Score in set train')
plt.plot(prevision_model_test, label = 'Score in set test')
plt.xlabel('random_state')
plt.ylabel('Score')
plt.title('INFLUENCE OF RANDOM_STATE ON THE SCORE')
plt.legend()
plt.show()

best_random_state = prevision_model_test.index(max(prevision_model_test))
best_score = max(prevision_model_test)
print('O melhor random_state é: {}; que gera um score de: {}'.format(best_random_state, best_score))

model_v1.score(x_train, y_train)
model_v1.score(x_test, y_test)


previson = model_v1.predict(x_test[110:120]) #y_test[110:120]
print(previson)