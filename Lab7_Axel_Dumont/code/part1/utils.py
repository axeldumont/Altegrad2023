"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2023
"""

import numpy as np


def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    ############## Task 1
    
    ##################
    # your code here #
    ##################

    X_train = np.zeros((n_train, max_train_card))
    y_train = np.zeros(n_train)
    for i in range(n_train):
        card = np.random.randint(1, max_train_card + 1)
        xi = np.random.choice(np.arange(1, 11), size=card, replace=False)
        yi = xi.sum()
        xi = np.concatenate(([0] * (max_train_card - card) , xi))
        X_train[i] = xi
        y_train[i] = yi
    return X_train, y_train


def create_test_dataset():
    
    ############## Task 2
    
    ##################
    # your code here #
    ##################
    X_test = []
    y_test = []

    nb_set = 10000
    max_cardinality = 100

    for card in range(5, max_cardinality+1, 5):

        X_i = np.zeros((nb_set, card))
        y_i = np.zeros(nb_set)
        
        for j in range(nb_set):
            card_i = np.random.randint(1, card + 1)
            xi = np.random.choice(np.arange(1, 11), size=card_i, replace=True)
            yi = np.unique(xi).sum()
            xi = np.concatenate(([0] * (card - card_i) , xi))
            X_i[j] = xi
            y_i[j] = yi
        
        X_test.append(X_i)
        y_test.append(y_i)

    return X_test, y_test