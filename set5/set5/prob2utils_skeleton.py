# Solution set for CS 155 Set 6, 2016/2017
# Authors: Fabian Boemer, Sid Murching, Suraj Nair

import numpy as np
import random
import math

def grad_U(Ui, Yij, Vj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """

    left = np.matmul(reg, Ui)
    right = np.dot(Ui, Vj)
    right2 = Vj * (Yij - right)
    return (left - right2) * eta
      

def grad_V(Vj, Yij, Ui, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    left = np.matmul(reg, Vj)
    right = np.dot(Ui, Vj)
    right2 = Ui * (Yij - right)
    return (left - right2) * eta

def get_err(U, V, Y, reg):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """

    error = 0
    U2 = np.transpose(U)
    V2 = np.transpose(V)

    for i in Y:
        error += (i[2] - np.dot(U2[i[0] - 1], V2[i[1] - 1])) ** 2

    left = (reg / 2) * (np.linalg.norm(U) ** 2 + np.linalg.norm(V) ** 2)

    return left + error/2


def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV^T)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """
    U = [[random.uniform(-0.5, 0.5) for q in range(M)] for p in range(K)]
    V = [[random.uniform(-0.5, 0.5) for t in range(N)] for z in range(K)]
    
    U = np.asarray(U, dtype = float)
    V = np.asarray(V, dtype = float)

    iter = 0
    loss = []
    loss.append(get_err(U,V,Y,reg))

    while iter < max_epochs:
        random.shuffle(Y)
        u_g = np.zeros((M,K))
        v_g = np.zeros((K,N))
        for i, j, y in Y:
            u_g[i - 1] = grad_U(np.tranpose(U)[i - 1], y, np.tranpose(V)[j - 1], reg, eta)
            np.transpose(v_g)[j - 1] = grad_V(np.tranpose(V)[j - 1], y, np.transpose(U[i - 1], reg, eta)

        U -= u_g
        V -= v_g
        loss.append(get_err(U,V,Y,reg))
        if (math.fabs(loss[-1] - loss[-2]) / math.fabs(loss[1] - loss[0]) <= eps):
            break
        iter += 1

    return U, V, loss
