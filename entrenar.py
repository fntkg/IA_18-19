import matplotlib.pyplot as plt #To drawn cool stuff
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
import k_fold_cross_validation
from termcolor import colored
import numpy as np

def train_classifiers(learner, data, labels, x_test, y_test):

    print("------K Fold Cross Validation------")
    print("Distribucion: ", learner)
    laplace = k_fold_cross_validation.k_fold_cross_validation(learner, 5, data, labels)

    print("------Evaluando sistema------")

    #Predict clases
    #Checkeamos el tipo de clasificador que queremos
    if learner == "Multinomial":
        classifier = MultinomialNB(alpha=laplace)
    elif learner == "Bernoulli":
        classifier = BernoulliNB(alpha=laplace)
    else:
        print(colored("Distribucion erronea", 'red'))

    #Entrenar clasificador
    classifier.fit(data,labels)

    #Con los emails_test, evaluar el sistema
    prediction = classifier.predict(x_test)
    prediction_prob = classifier.predict_proba(x_test)


    precision = metrics.accuracy_score(y_test, prediction)

    """metrics.plot_confusion_matrix(classifier, x_test, y_test, normalize='true')
    plt.show()


    metrics.plot_precision_recall_curve(classifier, x_test, y_test)
    plt.show()"""

    print("Sistema evaluado con exito")

    return classifier, precision, learner, laplace
