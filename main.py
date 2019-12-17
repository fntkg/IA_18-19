import load_mails
import entrenar

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt #To drawn cool stuff
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from termcolor import colored
import numpy as np

######################################################
# Auxiliar functions
######################################################
def bow(mails, y, mails_test, y_test):
    print("------Obtaining BOWs from emails-----")
    vectorizer  = CountVectorizer(ngram_range=(1, 1))  # Initialize BOW structure
    X = vectorizer.fit_transform(mails)                # BOW with word counts
    X_test = vectorizer.transform(mails_test)          # BOW (de enron6) with word counts

    #Transformar bolsa de palabras a bolsa de palabras con frecuencia de aparicion
    vectorizer_attempt = TfidfTransformer()
    X_X = vectorizer_attempt.fit_transform(X)
    print("BOW creado con exito")
    return X_X, X_test

def best_classifier(X_X, y, X_test, y_test):
    classifier1, accuracy1, learner1, laplace1 = entrenar.train_classifiers("Multinomial", X_X, y, X_test, y_test)
    classifier2, accuracy2, learner2, laplace2 = entrenar.train_classifiers("Bernoulli", X_X, y, X_test, y_test)

    if accuracy1 > accuracy2:
        precision = accuracy1
        classifier = classifier1
        learner = learner1
        laplace = laplace1
    else:
        precision = accuracy2
        classifier = classifier2
        learner = learner2
        laplace = laplace2

    ey = classifier.predict_proba(X_test)[:,1]
    prediction = classifier.predict(X_test)
    f1score =  metrics.f1_score(y_test, prediction)
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, ey)
    confusion_matriz = metrics.confusion_matrix(y_test, prediction)
    matriz_normalizada = metrics.confusion_matrix(y_test, prediction, normalize='true')

    ## TODO: Averiguar el valor umbral
    count = 0
    for i in precision:
        count = count + 1
        if i == 0.9879732739420936:
            print("Umbral: ", thresholds[count-1])
            break

    print(colored("------Mejor sistema de aprendizaje---", 'green'))
    print(">> Modelo: ", learner, "<<")
    print(">> Laplace: ", laplace, "<<")
    print("F1_score: ", f1score)
    print("Mejor umbral: ", thresholds)
    """print("Precision: ", precision)
    print("Recall: ", recall)
    print("Umbral: ", thresholds)"""
    print("Confusion matrix: \n", confusion_matriz)
    print("Normalized confusion matrix: \n", matriz_normalizada)
    print("-------------------------------------")

    #Dibujar (o no) las metricas
    """drawn = "a"

    while drawn != "yes" and drawn != "no":
        print("Desea dibujar la grafica precision_recall y la matriz de confusion?\nEscribe \"yes\" o \"no\"")
        drawn = input()
        if drawn != "yes" and drawn != "no":
            print(colored("No vayas de listo", 'red'))

    if drawn == "yes":
        metrics.plot_confusion_matrix(classifier, X_test, y_test, normalize='true')
        plt.show()


        metrics.plot_precision_recall_curve(classifier, X_test, y_test)
        plt.show()"""
######################################################
# Main
######################################################

mails, y, mails_test, y_test = load_mails.load_mails()

X_X, X_test = bow(mails, y, mails_test, y_test)

#LLamar a k_fold_cross_validation
best_classifier(X_X, y, X_test, y_test)
