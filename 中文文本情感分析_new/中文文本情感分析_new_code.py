
import numpy as np
import pandas as pd
import pretty_errors
pretty_errors.activate()

'''
先改这个！！！
'''
############################################
user1 = input("请输入你姓名首写小字母：")
############################################

if(user1=="rbq"):
    ori_data=r"D:\大一年度项目资料\中文文本情感分析_new\ori data.xlsx"
    stop_words_file = r"D:\大一年度项目资料\中文文本情感分析_new\哈工大停用词表.txt"
    data=r"D:\大一年度项目资料\中文文本情感分析_new\data.xlsx"
elif(user1=="sr"):
    ori_data=r"/Users/surui/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/cfa43d013fb378e45355d09126bc529f/Message/MessageTemp/bd32bd434566b9ad7bb1db6ed50133bf/File/ori data.xlsx"
    stop_words_file = r"中文文本情感分析_new/哈工大停用词表.txt"
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
data['cut_comment'] = data.comment.apply(chinese_word_cut)
#和这个等价
#data['cut_comment'] = data.content.apply(chinese_word_cut)

#data['cut_comment'] = data.iloc[:, 0].apply(chinese_word_cut)
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
test = pd.DataFrame(vect.fit_transform(X_train).toarray(), columns=vect.get_feature_names_out())
test.head()


# #### 训练模型

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

X_train_cleaned = X_train[y_train.notnull()]
y_train_cleaned = y_train[y_train.notnull()]

X_train_vect = vect.fit_transform(X_train_cleaned)
nb.fit(X_train_vect, y_train_cleaned)

# 保存模形--- 没有，先不保存
import joblib
joblib.dump(nb, 'model.pkl')

# #### 评估模型
train_score = nb.score(X_train_vect, y_train_cleaned)
print(train_score)

X_test_vect = vect.transform(X_test)

# 计算精确率
from sklearn.metrics import precision_score

precision = precision_score(y_test, nb.predict(X_test_vect), average='weighted')
print("精确率:", precision)

# 计算召回率
from sklearn.metrics import recall_score

recall = recall_score(y_test, nb.predict(X_test_vect), average='weighted')
print("召回率:", recall)


# 计算F1值
from sklearn.metrics import f1_score

f1 = f1_score(y_test, nb.predict(X_test_vect), average='weighted')
print("F1值:", f1)

# 计算准确率
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, nb.predict(X_test_vect))
print("准确率:", accuracy)



from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 计算真值
y_test_true = y_test
#打印什么真值，不打印，闹心
#print(y_test_true)
# 计算预测值
y_test_pred = nb.predict_proba(X_test_vect)[:,1]

# 计算ROC曲线
# 切换字符串为数字--先不整
y_test_true = y_test_true.replace({'1': 1, '0': 0})
#fpr, tpr, thresholds = roc_curve(y_test_true, y_test_pred)
#roc_auc = auc(fpr, tpr)

# 绘制ROC曲线---不画
def draw():
    plt.figure()
    #plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

#draw()



# #### 分析数据 


def ana():
    
    #data.head()
    
    data = pd.read_excel(data)#？
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
#先不分析
#ana()


