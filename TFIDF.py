# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')
def review_to_wordlist(review):
    '''
    把IMDB的评论转成词序列
    参考：http://blog.csdn.net/longxinchen_ml/article/details/50629613
    '''
    # 去掉HTML标签，拿到内容
    review_text = BeautifulSoup(review, "html.parser").get_text()
    # 用正则表达式取出符合规范的部分
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    # 小写化所有的词，并转成词list
    words = review_text.lower().split()
    # 返回words
    return words

train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
test = pd.read_csv('testData.tsv', header=0, delimiter="\t", quoting=3)
#print(train.head())
#print(test.head())

# 预处理数据
label = train['sentiment']
train_data = []
for i in range(len(train['review'])):
    train_data.append(' '.join(review_to_wordlist(train['review'][i])))
test_data = []
for i in range(len(test['review'])):
    test_data.append(' '.join(review_to_wordlist(test['review'][i])))

# 预览数据
#print (train_data[0], '\n')
#print( test_data[0])

from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF

tfidf = TFIDF(min_df=2, # 最小支持度为2
           max_features=None,
           strip_accents='unicode',
           analyzer='word',
           token_pattern=r'\w{1,}',
           ngram_range=(1, 3),  # 二元文法模型
           use_idf=1,
           smooth_idf=1,
           sublinear_tf=1,
           stop_words = 'english') # 去掉英文停用词

# 合并训练和测试集以便进行TFIDF向量化操作
data_all = train_data + test_data
len_train = len(train_data)

tfidf.fit(data_all)
data_all = tfidf.transform(data_all)
# 恢复成训练集和测试集部分
train_x = data_all[:len_train]
test_x = data_all[len_train:]
print ('TF-IDF over.')

from sklearn.cross_validation import cross_val_score
import numpy as np


#随机森林方法
from sklearn.ensemble import RandomForestClassifier
model_RF =  RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
model_RF.fit(train_x, label)

#输出结果
print ("Score(RF with 10-folds): ", np.mean(cross_val_score(model_RF, train_x, label, cv=10, scoring='roc_auc')))
test_predicted = np.array(model_RF.predict(test_x))
print ('Test result of RF saving...')
lr_output = pd.DataFrame(data=test_predicted, columns=['sentiment'])
lr_output['id'] = test['id']
lr_output = lr_output[['id', 'sentiment']]
lr_output.to_csv('~/Documents/sentiment/movie_sentiment/result/tfidf_RF_submission.csv', index=False)
print('the result of RF saved as RF_submission.csv.')

#朴素贝叶斯方法
from sklearn.naive_bayes import MultinomialNB as MNB

model_NB = MNB()
model_NB.fit(train_x, label)
MNB(alpha=1.0, class_prior=None, fit_prior=True)

from sklearn.cross_validation import cross_val_score
import numpy as np

print ("Score(Bayesian estimation with 10-folds): ", np.mean(cross_val_score(model_NB, train_x, label, cv=10, scoring='roc_auc')))
test_predicted = np.array(model_NB.predict(test_x))
print( 'Test result of Naive Bayes saving...')
nb_output = pd.DataFrame(data=test_predicted, columns=['sentiment'])
nb_output['id'] = test['id']
nb_output = nb_output[['id', 'sentiment']]
nb_output.to_csv('~/Documents/sentiment/movie_sentiment/result/tfidf_Naive_Bayes_submission.csv', index=False)
print ('the result of Naive Bayes saved as Naive_Bayes_submission.csv.')


#逻辑回归方法
from sklearn.linear_model import LogisticRegression as LR
from sklearn.grid_search import GridSearchCV
# 设定grid search的参数
grid_values = {'C':[30]}  
# 设定打分为roc_auc
model_LR = GridSearchCV(LR(dual = True, random_state = 0), grid_values, scoring = 'roc_auc', cv = 20)
model_LR.fit(train_x, label)
# 20折交叉验证
GridSearchCV(cv=20, estimator=LR(C=1.0, class_weight=None, dual=True,
             fit_intercept=True, intercept_scaling=1, penalty='L2', random_state=0, tol=0.0001),
        fit_params={}, iid=True, n_jobs=1,
        param_grid={'C': [30]}, pre_dispatch='2*n_jobs', refit=True,
        scoring='roc_auc', verbose=0)
#输出结果

print ("Score(Logistic Regression with 20-folds): ",model_LR.grid_scores_)
test_predicted = np.array(model_LR.predict(test_x))
print ('Test result of LogisticRegression saving...')
lr_output = pd.DataFrame(data=test_predicted, columns=['sentiment'])
lr_output['id'] = test['id']
lr_output = lr_output[['id', 'sentiment']]
lr_output.to_csv('~/Documents/sentiment/movie_sentiment/result/tfidf_LR_submission.csv', index=False)
print('the result of LR saved as LR_submission.csv.')




