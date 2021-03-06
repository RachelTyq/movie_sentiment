import numpy as np
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics  import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
def performance(y_true , predict , color = "g" , ann = True):
    acc = accuracy_score(y_true , predict[:,1] > 0.5)
    auc = roc_auc_score(y_true , predict[:,1])
    fpr , tpr , thr = roc_curve(y_true , predict[:,1])
    plt.figure()
    plt.plot(fpr , tpr )
 
df = pd.read_csv("labeledTrainData.tsv" , delimiter="\t") #导入数据 tsv是按照\t分割的
test = pd.read_csv('testData.tsv', header=0, delimiter="\t", quoting=3)
#print(df.head(50)) #查看数据存储结构
split = 0.7
d_train = df[:int(split * len(df))] #按照7:3的比例分为测试集和训练集
d_test = df[int((split) * len(df)) :]
#print(len(df))
#print(len(d_train))
#print(len(d_test))
vectorizer =  CountVectorizer(ngram_range=(2, 2), decode_error="ignore",
                                        token_pattern = r'\b\w+\b',min_df=1) #初始化单词计数向量器
features = vectorizer.fit_transform(d_train.review) #训练样本特征值
test_features = vectorizer.transform(d_test.review) #测试样本的特征值
test_real_features=vectorizer.transform(test.review)
i = 45000
j = 10
words = vectorizer.get_feature_names()[i:i+10]
#print(words)
#print(features[:3 , i:i+10].todense())
 
NBmodel = MultinomialNB()
NBmodel.fit(features , d_train.sentiment) #训练模型
predict_nb1 = NBmodel.predict_proba(test_features) #返回在每一类对应的概率
predict_nb2=NBmodel.predict_proba(test_real_features) #返回在每一类对应的概率

y_true_nb =  d_test.sentiment
predict_nb = predict_nb1
acc_nb = accuracy_score(y_true_nb, predict_nb[:, 1] > 0.5)
print("2-gram NB准确率为 = %f" % acc_nb)
auc_nb = roc_auc_score(y_true_nb, predict_nb[:, 1])
print("2-gram NB auc=%f"%auc_nb)

test_predicted_nb = np.array(predict_nb2[:, 1]>0.5)
a=test_predicted_nb+0;
nb_output = pd.DataFrame(data=a, columns=['sentiment'])
nb_output['id'] = test['id']
nb_output = nb_output[['id', 'sentiment']]
nb_output.to_csv("~/Documents/sentiment/movie_sentiment/result/2-gram-nb-submission.csv", index=False)




from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LR
LGmodel =LR()
LGmodel.fit(features , d_train.sentiment) #训练模型
predict_lr1 = LGmodel.predict_proba(test_features) #返回在每一类对应的概率
predict_lr2=LGmodel.predict_proba(test_real_features) #返回在每一类对应的概率
#print(predict1)
# performance(d_test.sentiment , predict1)
y_true_lr =  d_test.sentiment
predict_lr = predict_lr1
acc_lr = accuracy_score(y_true_lr, predict_lr[:, 1] > 0.5)
#print(predict[:,1])
print("2-gram LR准确率为 = %f" % acc_lr)
auc_lr = roc_auc_score(y_true_lr, predict_lr[:, 1])
print("2-gram LRauc=%f"%auc_lr)


test_predicted_lr = np.array(predict_lr[:, 1]>0.5)
a=test_predicted_lr+0;
lr_output = pd.DataFrame(data=a, columns=['sentiment'])
lr_output['id'] = test['id']
lr_output = lr_output[['id', 'sentiment']]
lr_output.to_csv("~/Documents/sentiment/movie_sentiment/result/2-gram-lr-submission.csv", index=False)

