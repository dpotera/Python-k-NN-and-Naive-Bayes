# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

from __future__ import division
import numpy as np
from scipy import spatial
from scipy.sparse import csc_matrix


def hamming_distance(X, X_train):
    """
    :param X: zbior porownwanych obiektow N1xD
    :param X_train: zbior obiektow do ktorych porownujemy N2xD
    Funkcja wyznacza odleglosci Hamminga obiektow ze zbioru X od
    obiektow X_train. ODleglosci obiektow z jednego i drugiego
    zbioru zwrocone zostana w postaci macierzy
    :return: macierz odleglosci pomiedzy obiektami z X i X_train N1xN2
    """
    return (spatial.distance.cdist(csc_matrix.toarray(X.astype(int)),
                                   csc_matrix.toarray(X_train.astype(int)), 'hamming') * X.shape[1]).astype(int)


def sort_train_labels_knn(Dist, y):
    """
    Funkcja sortujaca etykiety klas danych treningowych y
    wzgledem prawdopodobienstw zawartych w macierzy Dist.
    Funkcja zwraca macierz o wymiarach N1xN2. W kazdym
    wierszu maja byc posortowane etykiety klas z y wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist
    :param Dist: macierz odleglosci pomiedzy obiektami z X
    i X_train N1xN2
    :param y: wektor etykiet o dlugosci N2
    :return: macierz etykiet klas posortowana wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist. Uzyc algorytmu mergesort.
    """
    map_dist = np.vectorize(lambda x: y[x])
    return map_dist(np.argsort(Dist, kind='mergesort'))


def p_y_x_knn(y, k):
    """
    Funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla
    kazdej z klas dla obiektow ze zbioru testowego wykorzystujac
    klasfikator KNN wyuczony na danych trenningowych
    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najblizszuch sasiadow dla KNN
    :return: macierz prawdopodobienstw dla obiektow z X
    """
    return [[np.count_nonzero(np.array(yn[:k]) == i)/k for i in range(1, 5)] for yn in y]


def classification_error(p_y_x, y_true):
    """
    Wyznacz blad klasyfikacji.
    :param p_y_x: macierz przewidywanych prawdopodobienstw <class 'numpy.ndarray'>
    :param y_true: zbior rzeczywistych etykiet klas 1xN. <class 'numpy.ndarray'>
    Kazdy wiersz macierzy reprezentuje rozklad p(y|x)
    :return: blad klasyfikacji
    """
    return np.count_nonzero(y_true != np.array([len(p) - np.argmax(p[::-1]) for p in p_y_x])) / y_true.size


def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: zbior danych walidacyjnych N1xD
    :param Xtrain: zbior danych treningowych N2xD
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartosci parametru k, ktore maja zostac sprawdzone
    :return: funkcja wykonuje selekcje modelu knn i zwraca krotke (best_error,best_k,errors), gdzie best_error to najnizszy
    osiagniety blad, best_k - k dla ktorego blad byl najnizszy, errors - lista wartosci bledow dla kolejnych k z k_values
    """
    dist = hamming_distance(Xval, Xtrain)
    sorted_labels = sort_train_labels_knn(dist, ytrain)
    errors = [classification_error(p_y_x_knn(sorted_labels, k), yval) for k in k_values]
    return min(errors), k_values[np.argmin(errors)], errors


def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: etykiety dla dla danych treningowych 1xN
    :return: funkcja wyznacza rozklad a priori p(y) i zwraca p_y - wektor prawdopodobienstw a priori 1xM
    """
    return [np.count_nonzero(np.array(ytrain) == k) / len(ytrain) for k in np.array(range(1, 5))]


def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: dane treningowe NxD
    :param ytrain: etykiety klas dla danych treningowych 1xN
    :param a: parametr a rozkladu Beta
    :param b: parametr b rozkladu Beta
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(x|y) zakladajac, ze x przyjmuje wartosci binarne i ze elementy
    x sa niezalezne od siebie. Funkcja zwraca macierz p_x_y o wymiarach MxD.
    """
    xtrain = np.transpose(Xtrain.toarray())
    N, D = xtrain.shape
    return np.array([np.divide(np.add([np.sum(np.bitwise_and(xtrain[i], np.equal(ytrain, k))) for i in range(N)], a - 1),
                      np.sum(np.array(ytrain) == k) + a + b - 2) for k in range(1, 5)])


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: wektor prawdopodobienstw a priori o wymiarach 1xM
    :param p_x_1_y: rozklad prawdopodobienstw p(x=1|y) - macierz MxD
    :param X: dane dla ktorych beda wyznaczone prawdopodobienstwa, macierz NxD
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla kazdej z klas z wykorzystaniem klasyfikatora Naiwnego
    Bayesa. Funkcja zwraca macierz p_y_x o wymiarach NxM.
    """
    X = X.toarray()

    def calc(row):
        out = X * row
        out += np.negative(X) - np.negative(X) * row
        out = np.apply_along_axis(np.prod, arr=out, axis=1)
        return out

    def calc2(row, x2):
        return np.multiply(row, x2)

    def normalise(row):
        Z = 1 / np.sum(row)
        return np.multiply(row, Z)

    test = np.apply_along_axis(calc, axis=0, arr=np.array(p_x_1_y).transpose())
    test = np.apply_along_axis(calc2, axis=1, arr=test, x2=p_y)
    test = np.apply_along_axis(normalise, axis=1, arr=test)

    return test


def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: zbior danych treningowych N2xD
    :param Xval: zbior danych walidacyjnych N1xD
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrow a do sprawdzenia
    :param b_values: lista parametrow b do sprawdzenia
    :return: funkcja wykonuje selekcje modelu Naive Bayes - wybiera najlepsze wartosci parametrow a i b. Funkcja zwraca
    krotke (error_best, best_a, best_b, errors) gdzie best_error to najnizszy
    osiagniety blad, best_a - a dla ktorego blad byl najnizszy, best_b - b dla ktorego blad byl najnizszy,
    errors - macierz wartosci bledow dla wszystkich par (a,b)
    """
    a_len, b_len = int(len(a_values)), int(len(b_values))
    errors = np.array([classification_error(p_y_x_nb(estimate_a_priori_nb(ytrain), estimate_p_x_y_nb(Xtrain, ytrain,
        a_values[int(i / a_len)], b_values[int(i % b_len)]), Xval), yval) for i in range(a_len * b_len)])
    return (min(errors), a_values[int(round(np.argmin(errors) / len(b_values)))],
            b_values[np.argmin(errors) % len(b_values)], np.array(errors).reshape(len(a_values), len(b_values)))
