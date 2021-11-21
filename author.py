import re
import os
import time
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC
# from sklearn.linear_model import Perceptron
# from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score


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
                doc = ''
                if os.path.isfile(os.path.join(path_d, file)):
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


def getData():
    df_train_, df_train = readfile('C50/C50train/')
    df_test_, df_test = readfile('C50/C50test/')
    return df_train, df_test


def setupData(df_train, df_test):
    return df_train['text'], df_train['label'], df_test['text'], df_test['label']


def vectorize(x_train, x_test):
    vectorizer = CountVectorizer(binary=True)   # 73.24%
    # vectorizer = TfidfVectorizer(ngram_range=(1, 5), min_df=5)  # 71.04%
    # vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5)  # 70.44%
    # vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=5)  # 68.92%

    vectorizer.fit(x_train)
    return vectorizer.transform(x_train), vectorizer.transform(x_test)


def predictionScore(x_train, x_test, y_train, y_test):
    # Using a StandardScalar makes it worse

    svm = LinearSVC()     # 73.24%
    # svm = SGDClassifier()   # 65.28%
    # svm = Perceptron()    # 64.08%

    svm.fit(x_train, y_train)
    prediction = svm.predict(x_test)
    return prediction, accuracy_score(y_test, prediction)


def main():
    print("\n")
    start = time.time()

    df_train, df_test = getData()
    x_train, y_train, x_test, y_test = setupData(df_train, df_test)

    # todo - start
    x_train, x_test = vectorize(x_train, x_test)
    prediction, score = predictionScore(x_train, x_test, y_train, y_test)
    print("Accuracy:", score)
    print("normalized_mutual_info_score between predict and test:", normalized_mutual_info_score(y_test, prediction))
    # todo - end

    end = time.time()
    print("Total time elapsed: {:0.2f} seconds.".format(end - start))


if __name__ == '__main__':
    main()