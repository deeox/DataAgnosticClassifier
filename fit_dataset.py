'''
This code will be used for running the data through the classifiers, training them and then using then trained models to
predict the labels of user input text.

Module name: fit_dataset.py
Input: Dataset in .csv format for training the model, Unlabelled .csv file to predict labels
Output: .csv file with predicted labels, Accuracies and the time taken by the model
'''

# Importing the required models, modules and libraries
from time import time
import regex as re
from flask import Flask, render_template, request

# Importing the data pre-processing tools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Importing the Machine Learning Models
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.metrics import accuracy_score

# Importing Libraries and their models for Natural Language Processing
import spacy
from nltk.corpus import stopwords

stopwords = stopwords.words('english')
nlp = spacy.load('en_core_web_sm')
filename = " "

# Importing pickle which will be used for saving the models and loading them later
import pickle

# Routing to the the home HTML page of the flask app
app = Flask(__name__)


@app.route('/')
def firstpage():
    return render_template('dataset.html')


# Retrieving the CSV file through the input path given by the user
@app.route('/train', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        path = request.files.get('myFile')

        # Reading the CSV file and converting it into a pandas data-frame
        df = pd.read_csv(path, encoding="ISO-8859-1")

        # Reading the name for the file for the model that will be saved
        filename = request.form['filename']

        # Reading the names of the feature and label as strings
        str1 = request.form['feature']
        str2 = request.form['label']

        # Assigning the feature and label variables to the respective columns
        if str1 in list(df) and str2 in list(df):
            y = df[str2]
            X = df[str1]
        else:
            return render_template('nameError.html')
        '''
        # Removing the punctuations and HTTP links in the feature text input
        x = []
        for subject in X:
            result = re.sub(r"http\S+", "", subject)
            replaced = re.sub(r'[^a-zA-Z0-9 ]+', '', result)
            x.append(replaced)
        X = pd.Series(x)
        '''
        X = X.str.lower()

        # Optional use of Tokenization and Lemmatization using Natural Language Processing in SpaCy
        """
        texts = []
        for doc in X:
            doc = nlp(doc, disable=['parser', 'ner'])
            tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
            tokens = [tok for tok in tokens if tok not in stopwords]
            tokens = ' '.join(tokens)
            texts.append(tokens)

        X = pd.Series(texts)
        """

        # Splitting the data-set into 2 parts : Training data and Test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

        tfidfvect = TfidfVectorizer(ngram_range=(1, 1))
        X_train_tfidf = tfidfvect.fit_transform(X_train)

        # Fitting all the classification models one by one and recording their accuracies and execution times

        start = time()
        clf1 = LinearSVC()
        clf1.fit(X_train_tfidf, y_train)
        pred_SVC = clf1.predict(tfidfvect.transform(X_test))

        a1 = accuracy_score(y_test, pred_SVC)
        end = time()
        print("accuracy SVC: {} and time: {} s".format(a1, (end - start)))

        start = time()
        clf2 = LogisticRegression(n_jobs=-1, multi_class='multinomial', solver='newton-cg')
        clf2.fit(X_train_tfidf, y_train)
        pred_LR = clf2.predict(tfidfvect.transform(X_test))
        a2 = accuracy_score(y_test, pred_LR)
        end = time()
        print("accuracy LR: {} and time: {}".format(a2, (end - start)))

        start = time()
        clf3 = RandomForestClassifier(n_jobs=-1)

        clf3.fit(X_train_tfidf, y_train)
        pred = clf3.predict(tfidfvect.transform(X_test))
        a3 = accuracy_score(y_test, pred)
        end = time()
        print("accuracy RFC: {} and time: {}".format(a3, (end - start)))

        start = time()
        clf4 = MultinomialNB()

        clf4.fit(X_train_tfidf, y_train)
        pred = clf4.predict(tfidfvect.transform(X_test))
        a4 = accuracy_score(y_test, pred)
        end = time()
        print("accuracy MNB: {} and time: {}".format(a4, (end - start)))

        start = time()
        clf11 = SGDClassifier(n_jobs=-1)

        clf11.fit(X_train_tfidf, y_train)
        pred = clf11.predict(tfidfvect.transform(X_test))
        a11 = accuracy_score(y_test, pred)
        end = time()
        print("accuracy SGDC: {} and time: {}".format(a11, (end - start)))
        start = time()
        clf12 = SGDClassifier(n_jobs=-1)

        clf12.fit(X_train_tfidf, y_train)
        pred = clf12.predict(tfidfvect.transform(X_test))
        a12 = accuracy_score(y_test, pred)
        end = time()
        print("accuracy XGBC: {} and time: {}".format(a12, (end - start)))

        # Comparing the accuracies of all the models and then saving(dumping) the model with the highest accuracy using pickle for later use.

        acu_list = [a1, a2, a3, a4, a11, a12]
        max_list = max(acu_list)

        if max_list == a1:
            pickle.dump(clf1, open(filename + '_model', 'wb'))
        elif max_list == a2:
            pickle.dump(clf2, open(filename + '_model', 'wb'))
        elif max_list == a3:
            pickle.dump(clf3, open(filename + '_model', 'wb'))
        elif max_list == a4:
            pickle.dump(clf4, open(filename + '_model', 'wb'))
        elif max_list == a11:
            pickle.dump(clf11, open(filename + '_model', 'wb'))
        elif max_list == a12:
            pickle.dump(clf12, open(filename + '_model', 'wb'))

        pickle.dump(tfidfvect, open(filename + '_tfidfVect', 'wb'))

        return render_template("result.html", ac1=a1, ac2=a2, ac3=a3, ac4=a4, ac11=a11, ac12=a12)


# Routing to the predict page
@app.route('/connect', methods=['POST', 'GET'])
def connect():
    return render_template('predict.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        path_predict = request.files.get('myFile_predict')  # Retrieving the requested path of the unlabelled CSV file

        # Reading content of the unlabelled CSV file and converting it into a Pandas Data-frame
        df_1 = pd.read_csv(path_predict, encoding="ISO-8859-1")

        # Reading the input feature text

        str3 = request.form['feature_predict']

        # Reading the file of the saved model
        filename_2 = request.form['filename_2']

        # Retrieving the Column in the data-base with the the input text feature
        if str3 in list(df_1):
            Z = df_1[str3]
        else:
            return render_template('nameError.html')

        """
        blob = []
        for doc in Z:
            doc = nlp(doc, disable=['parser', 'ner'])
            tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
            tokens = [tok for tok in tokens if tok not in stopwords]
            tokens = ' '.join(tokens)
            blob.append(tokens)

        Z = pd.Series(blob)
        """

        # Loading the dumped models using pickle
        loaded_model = pickle.load(open(filename_2 + '_model', 'rb'))
        loaded_tfidf = pickle.load(open(filename_2 + '_tfidfVect', 'rb'))

        # Transforming the input text feature data using TF_IDF
        Z_predict = loaded_tfidf.transform(Z)

        # Predicting the results using the loaded model
        predict_model = loaded_model.predict(Z_predict)

        # Creating a column for the predicted labels
        df_1['label'] = predict_model

        # Requesting the name of the output CSV file followed by reading it
        name = request.form['csv_name']

        # Converting the final data-frame into the output CSV with the name requested
        df_1.to_csv(name, sep=',')

        # Making the dataframe into a 2-D array to pass it to HTML
        texts = df_1['text']
        labels = df_1['label']
        data_print = [[text, lab] for text, lab in zip(texts, labels)]

        return render_template("result1.html", data_print=data_print)


# Runs the flask app
if __name__ == '__main__':
    app.run(debug=True)
