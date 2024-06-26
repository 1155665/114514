
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
# #### 训练模型

min_length = 4
X_train_cleaned = X_train[X_train.str.len() >= min_length]
y_train_cleaned = y_train[X_train.str.len() >= min_length]
X_train_vect = vect.fit_transform(X_train_cleaned)

a = 4
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

# 保存
import joblib
joblib.dump(nb, 'model.pkl')

print("")
print("评估")
print("--------------------------------------------")

#评估模型
train_score = nb.score(X_train_vect, y_train_cleaned)
print("SCORE :",train_score)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

X_test_vect = vect.transform(X_test)
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

print("--------------------------------------------")
print("")


#from sklearn.metrics import roc_curve, auc,roc_auc_score
#from sklearn.metrics import 

import matplotlib.pyplot as plt

# 计算真值
y_test_true = y_test
#打印什么真值，不打印，闹心
#print(y_test_true)
# 计算预测值
if(a==1 or a==2):
    y_test_pred = nb.predict_proba(X_test_vect)
else:
    y_test_pred = nb.decision_function(X_test_vect)

y_test = y_test_true.replace({'1': 1.0, '0': 0.0,'2':2.0})
ytest_l = list(np.array(y_test))
ytest_one = label_binarize(ytest_l, classes=[0,1,2]) 
#ytest_one = ytest_l

'''宏平均法'''
#CC 4.0 BY-SA article：131918191
#begin
macro_AUC = {}
macro_FPR = {}
macro_TPR = {}
# 获的每一个类别对应的TPR、FPR、AUC
for i in range(ytest_one.shape[1]):
    macro_FPR[i],macro_TPR[i],thresholds = metrics.roc_curve(ytest_one[:,i], y_test_pred[:,i])
    macro_AUC[i] = metrics.auc(macro_FPR[i],macro_TPR[i])
print("macro_AUC :",macro_AUC)
 
# 把所有的FPR合并去重、排序
macro_FPR_final = np.unique(np.concatenate([macro_FPR[i] for i in range(ytest_one.shape[1])]))
 
# 在每个类别中计算macro_FPR_final对应的TPR 并相加求平均
macro_TPR_all = np.zeros_like(macro_FPR_final)
for i in range(ytest_one.shape[1]):
    macro_TPR_all = macro_TPR_all + np.interp(macro_FPR_final, macro_FPR[i], macro_TPR[i])
macro_TPR_final = macro_TPR_all / ytest_one.shape[1] # 注：当FPR对应多个TPR时，interp会返回最大的那个TPR
macro_AUC_final = metrics.auc(macro_FPR_final, macro_TPR_final)
 
'''画图'''
plt.figure(figsize=(8,6))
plt.plot(macro_FPR[0],macro_TPR[0],'b.-', label='1ROC  AUC={:.2f}'.format(macro_AUC[0]), lw=2)
plt.plot(macro_FPR[1],macro_TPR[1],'y.-', label='2ROC  AUC={:.2f}'.format(macro_AUC[1]), lw=2)
plt.plot(macro_FPR[2],macro_TPR[2],'r.-', label='3ROC  AUC={:.2f}'.format(macro_AUC[2]), lw=2)
plt.plot(macro_FPR_final,macro_TPR_final,'kx-', label='macroROC  AUC={:.2f}'.format(macro_AUC_final), lw=2)

plt.plot([0,1], [0,1], 'k--', label='random classifier') 
plt.xlabel('FPR',fontsize=13)
plt.ylabel('TPR',fontsize=13)

if(a==1):
    at = "MultinomialNB"
elif(a==2):
    at = "RandomForestClassifier"
elif(a==3):
    at = "LogisticRegression"
elif(a==4):
    at = "SVM"

plt.title(at,fontsize=13)
plt.grid(linestyle='-.')
plt.legend(loc='lower right',framealpha=0.8, fontsize=8)
plt.show()

#end



'''

# #### 标数据 


def ana(d):
    #data.head()
    data = pd.read_excel(d)#？
    #data = pd.read_excel("data.xlsx")#？？这是啥  #错了错了,sorry
    #data['cut_comment'] = data.comment.apply(chinese_word_cut)#

    X=data['cut_comment']
    X_vec = vect.transform(X)
    nb_result = nb.predict(X_vec)

    #predict_proba(X)[source] 返回概率
    data['nb_result'] = nb_result
    test = pd.DataFrame(vect.fit_transform(X).toarray(), columns=vect.get_feature_names_out())
    test.head()
    data.to_excel("data_result.xlsx",index=False)
    #return ?
#先不分析
#ana(data)

'''