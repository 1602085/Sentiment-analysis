import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # transform text to feature vector
import numpy as np
from sklearn.neural_network import MLPClassifier    # multilayer perceptron classifier, neural network for classification


df = pd.read_csv(r'/home/shikha/Desktop/reviews_train.csv', error_bad_lines=False, sep='delimiter', header=None, engine='python')
#print(df.head())


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
print(cv)

x_train = cv.fit_transform(g) # learn vocabulary and idf from training set 
#print(x_train)

fe1 = x_train.toarray()
print(fe1)
print(len(fe1))
print(len(fe1[0]))
fe = fe1.tolist()
#print(fe)

sample, feature = fe1.shape

def perceptron(feature_matrix, label, T=50):
    theta = np.zeros(feature)
    theta0 = 0.0
    for t in range(T):
        for i in range(sample):
            if label[i] * (np.dot(feature_matrix[i], theta) + theta0) <= 0:
               theta += label[i] * feature_matrix[i]
               theta0 += label[i]
    return theta, theta0

label = np.array(q)
theta, theta0 = perceptron(fe1, label, T=50)



df1 = pd.read_csv(r'/home/shikha/Desktop/reviews_test.csv', error_bad_lines=False, sep='delimiter', header=None, engine='python')
#print(df1.head())

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

print(len(c))
cv1 = TfidfVectorizer(min_df=1, stop_words='english')
x_test = cv1.fit_transform(a)
fe2 = x_test.toarray()

fe3 = np.zeros([500, 8815])
fe4 = np.concatenate([fe2, fe3], axis=1)


print(len(fe4))
print(len(fe4[0]))
fee = fe4.tolist()


clf = MLPClassifier(hidden_layer_sizes = (100, 50, 50, 2), max_iter = 400)  # neural network for classification

trained = clf.fit(fe, q) # fit the vocabulary and sentiment
res = trained.predict(fee) # predict on test data 

res =  res.tolist()   

total=0
for i in range(len(c)):
    total += c[i]  

print(total)


sample1, feature1 = fe4.shape
label1 = np.array(c)


def classify(feature_matrix1, theta, theta0):
    predictions = np.zeros(sample1)
    for i in range(sample1):
        prediction = np.dot(theta, feature_matrix1[i]) + theta0
        if (prediction > 0):
            predictions[i] = 1
        else:
            predictions[i] = -1
    return predictions


res1 = classify(fee, theta, theta0)
print(res1)
print(label1)

def accuracy(label1, res):
    return (label1 == res).mean()
   

acc = accuracy(label1, res1)
print(acc)





