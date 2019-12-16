import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # transform text to feature vector
import numpy as np
from sklearn.neural_network import MLPClassifier    # multilayer perceptron classifier, neural network for classification


df = pd.read_csv(r'/home/shikha/Downloads/reviews_train.csv', error_bad_lines=False, sep='delimiter', header=None, engine='python')
print(df.head())


g=[]  # for storing review text of users
h=[]  # for storing sentiment
q=[]  # for storing sentiment in integer


n=len(df)


for i in range(1,n):
    z=df.iloc[i][0]
    p=z.split('\t')
    g.extend(p[4:5])
    h.extend(p[0:1])


for j in h:
    q.append(int(j))


cv = TfidfVectorizer(min_df=1, stop_words='english')  # performs the TF-IDF transformation from a provide matrix of counts that score for identical words and common words value set to zeros

x_train = cv.fit_transform(g) # learn vocabulary and idf from training set 


fe1 = x_train.toarray()
fe = fe1.tolist()


df1 = pd.read_csv(r'/home/shikha/Downloads/reviews_test.csv', error_bad_lines=False, sep='delimiter', header=None, engine='python')
print(df1.head())

a=[]
b=[]
c=[]


n1=len(df1)
for o in range(1,n1):
    z1=df1.iloc[o][0]
    p1=z1.split('\t')
    a.extend(p1[4:5])
    b.extend(p1[0:1])


for f in b:
    c.append(int(f))


cv1 = TfidfVectorizer(min_df=1, stop_words='english')
x_test = cv1.fit_transform(a)
fe2 = x_test.toarray()
fee = fe2.tolist()


clf = MLPClassifier(hidden_layer_sizes = (100, 50, 50, 2), max_iter = 400)  # neural network for classification

trained = clf.fit(fe, q) # fit the vocabulary and sentiment
res = trained.predict(fee) # predict on test data 
res =  res.tolist()   

print(res)

total=0
for i in range(len(c)):
    total += c[i]  

def accuracy(c, res):
    pred = 0
    for i in range(len(c)):
        if c[i] == res[i]:
           pred += c[i]
    accur = pred / total
    return accur

score = accuracy(c, res) # calculate accuracy of system
print(score)




