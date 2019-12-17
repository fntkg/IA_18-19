from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
import numpy as np

def k_fold_cross_validation(learner, folds, data, labels):
    best_laplace = 0 #Devuele el mejor valor de Laplace
    best_f1_score = 0.0
    best_accuracy = 0.0

    labels_A = np.array(labels)

    print("Probando diferentes suavizados de laplace...")

    for i in range(1,10):
    # i = valor de laplace
        _f1_score = 0.0
        score = 0.0
        accuracy = 0.0

        kf = KFold(n_splits=folds)

        for training_index, test_index in kf.split(data):

            #Creamos las nuevas particiones de datos
            training_data = data[training_index]
            training_labels = labels_A[training_index]
            validation_data = data[test_index]
            validation_labels = labels_A[test_index]

            #Checkeamos el tipo de clasificador que queremos
            if learner == "Multinomial":
                classifier = MultinomialNB(alpha=i)
            else:
                classifier = BernoulliNB(alpha=i)

            #Entrenar modelo
            classifier.fit(training_data, training_labels)

            #Predecir las clases de los emails con los datos de validacion
            prediction = classifier.predict(validation_data)

            score =  metrics.f1_score(validation_labels, prediction)
            #f1_score (valores reales, valores obtenidos por el classifier)
            _f1_score = _f1_score + score
            accuracy = accuracy + metrics.accuracy_score(validation_labels, prediction)

        #Calcular la media del score y del accuracy y quedarnos con la mejor
        f1_score_mean = _f1_score/folds
        accuracy_mean = accuracy/folds
        if f1_score_mean > best_f1_score:
            best_f1_score = f1_score_mean
        if accuracy_mean > best_accuracy:
            best_accuracy = accuracy_mean
            best_laplace = i

    """#Presentar resultados por pantalla
    print("Clasificador: ", learner)
    print("Mejor valor de Laplace: ", best_laplace)
    print("Mejor f1_score: ", best_f1_score)
    print("Mejor precicion: ", best_accuracy)"""

    return best_laplace
