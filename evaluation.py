import pandas as pd
from rich import traceback,print
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc
import warnings
traceback.install()
warnings.filterwarnings("ignore")
# import matplotlib.pyplot as plt

df = pd.read_excel('output copy.xlsx')

column2 = df.iloc[:, 1]
column3 = df.iloc[:, 2]

column2 = column2.replace({0: 'neutral', 1: 'positive', 2: 'negative'})
####################
column2 = column2.replace({'neutral': -1,'positive': 0, 'negative':1 })
column3 = column3.replace({'neutral': -1,'positive': 0, 'negative':1 , 'Error': 0})
y_test = column2
y_pred = column3

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


print("")
print("Evaluation")
print("--------------------------------------------")
precision = precision_score(y_test, y_pred,average='weighted')
recall = recall_score(y_test, y_pred,average='weighted')
print("Precision:", precision)
print("Recall   :", recall)
f1 = 2 * (precision * recall) / (precision + recall)
print("F1 Score :", f1)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy :", accuracy)
print("--------------------------------------------")
print("")



