# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup

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




#Word2vec

import gensim
import nltk
from nltk.corpus import stopwords

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_wordlist( review, remove_stopwords=False ):
    review_text = BeautifulSoup(review, "html.parser").get_text()
    review_text = re.sub("[^a-zA-Z]"," ", review_text)

    words = review_text.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return(words)

def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    '''
    将评论段落转换为句子，返回句子列表，每个句子由一堆词组成
    '''
    raw_sentences = tokenizer.tokenize(review.strip())

    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            # 获取句子中的词列表
            sentences.append( review_to_wordlist( raw_sentence, remove_stopwords ))
    return sentences
print (len(train_data)) 
sentences = []
for i, review in enumerate(train["review"]):
    sentences += review_to_sentences(review, tokenizer)


unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)
print ('预处理unlabeled_train data...')
#print (len(train_data)) 
#print (len(sentences))
#构建word2vec模型
import time
from gensim.models import Word2Vec
# 模型参数
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words


# 训练模型
print("训练模型中...")
model = Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

print( '保存模型...')
model.init_sims(replace=True)
model_name = "300features_40minwords_10context"
model.save(model_name)

#预览模型
print(model.doesnt_match("man woman child kitchen".split()))
print(model.doesnt_match("france england germany berlin".split()))
print(model.doesnt_match("paris berlin london austria".split()))
print(model.most_similar("man"))
print(model.most_similar("queen"))
print(model.most_similar("awful"))



def makeFeatureVec(words, model, num_features):
    '''
    对段落中的所有词向量进行取平均操作
    '''
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0.

    # Index2word包含了词表中的所有词，为了检索速度，保存到set中
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])

    # 取平均
    featureVec = np.divide(featureVec, nwords)
    return featureVec
#使用Word2vec特征

def getAvgFeatureVecs(reviews, model, num_features):
    '''
    给定一个文本列表，每个文本由一个词列表组成，返回每个文本的词向量平均值
    '''
    counter = 0

    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")

    for review in reviews:
       if counter % 5000 == 0:
           print("Review %d of %d" % (counter, len(reviews)))

       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)

       counter = counter + 1
    return reviewFeatureVecs

trainDataVecs = getAvgFeatureVecs(train_data, model, num_features)
testDataVecs = getAvgFeatureVecs(test_data, model, num_features)

#高斯贝叶斯+Word2vec训练
from sklearn.naive_bayes import GaussianNB as GNB

model_GNB = GNB()
model_GNB.fit(trainDataVecs, label)

from sklearn.cross_validation import cross_val_score
import numpy as np

print ("高斯贝叶斯分类器10折交叉验证得分: ", np.mean(cross_val_score(model_GNB, trainDataVecs, label, cv=10, scoring='roc_auc')))

result = model_GNB.predict( testDataVecs )

output1 = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output1.to_csv( "~/Documents/sentiment/movie_sentiment/result/gnb_word2vec.csv", index=False, quoting=3 )


#随机森林+Word2vec训练

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier( n_estimators = 100, n_jobs=2)

print("Fitting a random forest to labeled training data...")
forest = forest.fit( trainDataVecs, label )
print ("随机森林分类器10折交叉验证得分: ", np.mean(cross_val_score(forest, trainDataVecs, label, cv=10, scoring='roc_auc')))

# 测试集
result = forest.predict( testDataVecs )

output2 = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output2.to_csv( "~/Documents/sentiment/movie_sentiment/result/rf_word2vec.csv", index=False, quoting=3 )

#深度神经网络+Word2vec训练
from sklearn.neural_network import MLPClassifier

MLP = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50, 50), random_state=5)

print("Fitting a MLP to labeled training data...")
MLP = MLP.fit( trainDataVecs, label )
print ("深度神经网络分类器10折交叉验证得分: ", np.mean(cross_val_score(MLP, trainDataVecs, label, cv=10, scoring='roc_auc')))

# 测试集
result = MLP.predict( testDataVecs )

output3 = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output3.to_csv( "~/Documents/sentiment/movie_sentiment/result/mlp_word2vec.csv", index=False, quoting=3 )


