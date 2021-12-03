import re
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC

def readfile(path):
    dirs = os.listdir(path)
    column_names = ['label', 'text']
    data_ = []
    data = []
    for d in dirs:
        if os.path.isdir(os.path.join(path, d)):
            path_d = path + d + '/'
            files = os.listdir(path_d)
            docs = ''
            for file in files:
                if os.path.isfile(os.path.join(path_d, file)):
                    doc = ''
                    f = open(os.path.join(path_d, file),'r')
                    for line in f:
                        if len(line)>0:
                            line = re.sub('[^a-zA-Z]', ' ', line)
                            words = []
                            for word in line.split():
                                if len(word)>1:
                                    word = word.strip().lower()
                                    words.append(word)
                            line = ' '.join(words)
                            doc = doc + " " + line
                row = [d, doc.strip()]
                docs = docs + " " + doc
                data.append(row)
            row_ = [d, docs.strip()]
            data_.append(row_)
    df_ = pd.DataFrame(data_, columns=column_names)
    df = pd.DataFrame(data, columns=column_names)
    return df_, df
                            
path_train = 'data/C50/C50train/'
path_test = 'data/C50/C50test/'   
df_train_, df_train = readfile(path_train)
df_test_, df_test = readfile(path_test)

df_train_.shape, df_train.shape, df_test_.shape, df_test.shape
len(df_train_.iloc[0, 1]), len(df_train.iloc[0, 1]), len(df_test_.iloc[0, 1]), len(df_test.iloc[0,1])
df_train_[:3]
df_train_[:3]
df_test_[:3]
df_test_[:3]
labels = list(df_train_['label'])
X_train = df_train['text']
y_train = df_train['label']
X_test = df_test['text']
y_test = df_test['label']

vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2), min_df=5)
vectorizer.fit(X_train)
X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# svm = LinearSVC(max_iter=10000)
# svm = LogisticRegression(max_iter=10000)
svm = LinearRegression()
# svm = MultinomialNB()

prob = svm.fit(X_train_vec, y_train)
y_pred_svm = svm.predict(X_test_vec)


print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(pd.DataFrame(confusion_matrix(y_test, y_pred_svm)).loc[10:25, 10:25])
print("normalized_mutual_info_score between predict and test:", normalized_mutual_info_score(y_test, y_pred_svm))

