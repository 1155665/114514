
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from rich import traceback,print
from sklearn import metrics
from sklearn.preprocessing import label_binarize
traceback.install()
import warnings
warnings.filterwarnings("ignore")

'''
先改这个！！！
'''
############################################
user1 = "sr"
############################################

if(user1=="rbq"):
    ori_data=r"D:\大一年度项目资料\中文文本情感分析_new\ori data.xlsx"
    stop_words_file = r"D:\大一年度项目资料\中文文本情感分析_new\哈工大停用词表.txt"
    data=r"D:\大一年度项目资料\中文文本情感分析_new\data.xlsx"
elif(user1=="sr"):
    ori_data=r"/Users/surui/Desktop/ori data 1.xlsx"
    stop_words_file = r"朴素贝叶斯/哈工大停用词表.txt"
    data=r"data.xlsx"
elif(user1=="hjm"):
    ori_data=r""
    stop_words_file = r""
    data=r"" 


data = pd.read_excel(ori_data).astype(str)
data.head()


#根据需要做处理
#去重、去除停用词
# #### jieba分词


import jieba

def chinese_word_cut(mytext):
    return" ".join(jieba.cut(mytext))



#更改一下，只让他分第一列--surui，将jieba分词的结果保存到data['cut_comment']中
#和这个等价，此文件第一列是comment，反正是第一列，注意你的文本是第几列
#data['cut_comment'] = data.content.apply(chinese_word_cut)
data['cut_comment'] = data.iloc[:, 0].apply(chinese_word_cut)
data.head()

# #### 提取特征
from sklearn.feature_extraction.text import CountVectorizer

def get_custom_stopwords(stop_words_file):
    with open(stop_words_file) as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    custom_stopwords_list = [i for i in stopwords_list]
    return custom_stopwords_list


stopwords = get_custom_stopwords(stop_words_file)

vect = CountVectorizer(max_df = 0.8, 
                       min_df = 3, 
                       token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b', 
                       stop_words=stopwords)


#划分数据集
X = data['cut_comment']
y = data.sentiment

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

#特征展示
#test = pd.DataFrame(vect.fit_transform(X_train).toarray(), columns=vect.get_feature_names_out())
#test.head()

# #### 训练模型

min_length = 3
X_train_cleaned = X_train[X_train.str.len() >= min_length]
y_train_cleaned = y_train[X_train.str.len() >= min_length]
X_train_vect = vect.fit_transform(X_train_cleaned)

a = 1
if(a==1):
    from sklearn.naive_bayes import MultinomialNB
    nb = MultinomialNB()
    nb.fit(X_train_vect, y_train_cleaned)
if(a==2):
    from sklearn.ensemble import RandomForestClassifier
    nb = RandomForestClassifier()
    nb.fit(X_train_vect, y_train_cleaned)
if(a==3):
    from sklearn.linear_model import LogisticRegression
    nb = LogisticRegression()
    nb.fit(X_train_vect, y_train_cleaned)
if(a==4):
    from sklearn.svm import SVC
    nb = SVC()
    nb.fit(X_train_vect, y_train_cleaned)



print("")
print("评估")
print("--------------------------------------------")


X_test_vect = vect.transform(X_test)

#评估模型
train_score = nb.score(X_train_vect, y_train_cleaned)
print("SCORE :",train_score)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 计算精确率
precision = precision_score(y_test, nb.predict(X_test_vect), average='weighted')
print("精确率:", precision)
# 计算召回率
recall = recall_score(y_test, nb.predict(X_test_vect), average='weighted')
print("召回率:", recall)
# 计算F1值
f1 = f1_score(y_test, nb.predict(X_test_vect), average='weighted')
print("F1值  :", f1)
# 计算准确率
accuracy = accuracy_score(y_test, nb.predict(X_test_vect))
print("准确率:", accuracy)

import matplotlib.pyplot as plt

# Binarize the labels
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_bin.shape[1]

# Compute the predicted probabilities for each class

if(a==1 or a==2):
    y_score = nb.predict_proba(X_test_vect)
elif(a==3|a==4):
    y_score = nb.decision_function(X_test_vect)

# Compute the ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# Plot the ROC curve for each class
plt.figure()
colors = ['blue', 'red', 'green', 'orange', 'purple']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

# Plot the random guessing line
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

# Set the plot title and labels
plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

# Show the plot
plt.show()

# #### 标数据 
'''

data = pd.read_excel(r"/Users/surui/Desktop/data.xlsx")
data.head()

data['cut_comment'] = data.comment.apply(chinese_word_cut)
data['cut_comment'] = data['cut_comment'].replace({"问题描述：":" "})
X=data['cut_comment']

X_vec = vect.transform(X) 
nb_result = nb.predict(X_vec)
#predict_proba(X)[source] 返回概率
data['nb_result'] = nb_result


test = pd.DataFrame(vect.fit_transform(X).toarray(), columns=vect.get_feature_names_out())
test.head()

data.to_excel("data_result.xlsx",index=False)



X = ["这位患者去死吧！！！"]
X_vec = vect.transform(X) 
nb_result = nb.predict(X_vec)
print(nb_result)


#print(nb.predict("你去死吧！！！".transform()))、

data1=pd.read_excel(r"/Users/surui/Desktop/data_result.xlsx")
'''