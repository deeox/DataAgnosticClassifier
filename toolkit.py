from time import time
import regex as re
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
from nltk.corpus import stopwords
from sklearn.naive_bayes import GaussianNB
import pickle

stopwords = stopwords.words('english')
nlp = spacy.load('en_core_web_sm')
filename = " "
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model.perceptron import Perceptron
from sklearn.linear_model.ridge import RidgeClassifierCV
from sklearn.linear_model.stochastic_gradient import SGDClassifier


names = ["LinearSVC", "LogisticRegression", "RandomForestClassifier", "MultinomialNB",
         "GaussianNB", "LogisticRegressionCV", "AdaBoostClassifier", "BernoulliNB",
         "Perceptron", "RidgeClassifierCV", "SGDClassifier", "XGBClassifier"]

classifiers = [
    LinearSVC(),
    LogisticRegression(n_jobs=-1, multi_class='multinomial', solver='newton-cg'),
    RandomForestClassifier(n_jobs=-1),
    MultinomialNB(),
    GaussianNB(),
    LogisticRegressionCV(n_jobs=-1),
    AdaBoostClassifier(),
    BernoulliNB(),
    Perceptron(n_jobs=-1),
    RidgeClassifierCV(),
    SGDClassifier(n_jobs=-1),
    XGBClassifier(n_jobs=-1)]

def read_csv(path):
    data = pd.read_csv(path, sep=',', encoding="ISO-8859-1")
    return data

def train_test_split(df, test_size):
    X = df.iloc[:, 0].copy()
    y = df.iloc[:, 1].copy()
    return train_test_split(X, y, test_size = test_size, random_state = 42)

def textProc_fit_transform(X):
    global tfidf = TfidfVectorizer(stop_words='english')
    X_tfidf = tfidf.fit_transform(X)
    return X_tfidf

def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

def fit_and_evaluate(X_train, X_test, y_train, y_test, clf):
    model = clf
    model.fit(X_train, y_train)

    model_pred = model.predict(X_test)
    model_mae = mae(y_test, model_pred)

    return model_mae



