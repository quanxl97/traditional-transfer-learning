# encoding=utf-8
"""
    Created on 9:52 2018/11/14 
    @author: Jindong Wang
"""

import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors
from sklearn import svm
from sklearn import preprocessing
import os
from numpy.matlib import repmat


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(
                np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, None, gamma)
    return K


def proxy_a_distance(source_X, target_X):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    train_X = np.vstack((source_X, target_X))
    train_Y = np.hstack((np.zeros(nb_source, dtype=int),
                         np.ones(nb_target, dtype=int)))

    clf = svm.LinearSVC(random_state=0)
    clf.fit(train_X, train_Y)
    y_pred = clf.predict(train_X)
    error = metrics.mean_absolute_error(train_Y, y_pred)
    dist = 2 * (1 - 2 * error)
    return dist


def estimate_mu(_X1, _Y1, _X2, _Y2):
    adist_m = proxy_a_distance(_X1, _X2)
    C = len(np.unique(_Y1))
    epsilon = 1e-3
    list_adist_c = []
    for i in range(1, C + 1):
        ind_i, ind_j = np.where(_Y1 == i), np.where(_Y2 == i)
        Xsi = _X1[ind_i[0], :]
        Xtj = _X2[ind_j[0], :]
        adist_i = proxy_a_distance(Xsi, Xtj)
        list_adist_c.append(adist_i)
    adist_c = sum(list_adist_c) / C
    mu = adist_c / (adist_c + adist_m)
    if mu > 1:
        mu = 1
    if mu < epsilon:
        mu = 0
    return mu


class BDA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, mu=0.5, gamma=1, T=10, mode='BDA', estimate_mu=False):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param mu: mu. Default is -1, if not specificied, it calculates using A-distance
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        :param mode: 'BDA' | 'WBDA'
        :param estimate_mu: True | False, if you want to automatically estimate mu instead of manally set it
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.mu = mu
        self.gamma = gamma
        self.T = T
        self.mode = mode
        self.estimate_mu = estimate_mu

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform and Predict using 1NN as JDA paper did
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_pred, list_acc
        '''
        list_acc = []
        X = np.hstack((Xs.T, Xt.T))
        X = X / np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        C = len(np.unique(Ys))
        H = np.eye(n) - 1 / n * np.ones((n, n))
        mu = self.mu
        M = 0
        Y_tar_pseudo = None
        Xs_new = None
        for t in range(self.T):
            N = 0
            M0 = e * e.T * C
            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
                for c in range(1, C + 1):
                    e = np.zeros((n, 1))
                    Ns = len(Ys[np.where(Ys == c)])
                    Nt = len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])

                    if self.mode == 'WBDA':
                        Ps = Ns / len(Ys)
                        Pt = Nt / len(Y_tar_pseudo)
                        alpha = Pt / Ps
                        mu = 1
                    else:
                        alpha = 1

                    tt = Ys == c
                    e[np.where(tt == True)] = 1 / Ns
                    yy = Y_tar_pseudo == c
                    ind = np.where(yy == True)
                    inds = [item + ns for item in ind]
                    e[tuple(inds)] = -alpha / Nt
                    e[np.isinf(e)] = 0
                    N = N + np.dot(e, e.T)

            # In BDA, mu can be set or automatically estimated using A-distance
            # In WBDA, we find that setting mu=1 is enough
            if self.estimate_mu and self.mode == 'BDA':
                if Xs_new is not None:
                    mu = estimate_mu(Xs_new, Ys, Xt_new, Y_tar_pseudo)
                else:
                    mu = 0
            M = (1 - mu) * M0 + mu * N
            M = M / np.linalg.norm(M, 'fro')
            # M = M / np.linalg.norm(M, axis=0)
            # Z = Z / np.linalg.norm(Z, axis=0)  # 归一化
            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = np.linalg.multi_dot(
                [K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = np.dot(A.T, K)
            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

            clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
            clf.fit(Xs_new, Ys.ravel())
            Y_tar_pseudo = clf.predict(Xt_new)
            acc = sklearn.metrics.accuracy_score(Yt, Y_tar_pseudo)
            list_acc.append(acc)
            print('{} iteration [{}/{}]: Acc: {:.4f}'.format(self.mode, t + 1, self.T, acc))
        return acc, Y_tar_pseudo, list_acc




if __name__ == '__main__':
    # office_caltech10_surf dataset
    domains_surf = ['\\amazon_SURF_L10.mat', '\\webcam_SURF_L10.mat', '\\dslr_SURF_L10.mat', '\\Caltech10_SURF_L10.mat']
    # office_caltech10_surf_zscore dataset
    domains_surf_zscore = ['\\amazon_zscore_SURF_L10.mat', '\\webcam_zscore_SURF_L10.mat',
                           '\\dslr_zscore_SURF_L10.mat', '\\Caltech10_zscore_SURF_L10.mat']
    # MNIST_USPS_SURF dataset
    domains_MU = ['\\MNIST_SURF.mat', '\\USPS_SURF.mat']
    # COIL20_SURF dataset
    domains_COIL20 = ['\\COIL1_SURF.mat', '\\COIL2_SURF.mat']
    # PIE_SURF dataset
    domains_pie = ['\\pie05_surf', '\\pie07_surf', '\\pie09_surf', '\\pie27_surf', '\\pie29_surf']

    # i = 3
    # j = 2
    for i in range(2):
        for j in range(2):
            if i != j:
                datapath = os.getcwd()  # 获取当前路径，路径不包含文件名
                # dataset: domains_surf, domains_surf_zscore, domains_MU, domains_COIL20
                # domains_pie, domains_office31_fc6, domains_office31_fc7
                # office-caltech10_surf dataset
                src = datapath + '\\data' + domains_surf[i]
                tar = datapath + '\\data' + domains_surf[j]
                # load data 数据集中每行表示一个样本
                src_domain = scipy.io.loadmat(src)
                tar_domain = scipy.io.loadmat(tar)

                Xs = src_domain['fts']
                Ys = src_domain['labels']
                Xt = tar_domain['fts']
                Yt = tar_domain['labels']

                # Z-Score标准化
                Xs = Xs / repmat(np.mat(Xs.sum(axis=1)).T, 1, Xs.shape[1])
                Xt = Xt / repmat(np.mat(Xt.sum(axis=1)).T, 1, Xt.shape[1])
                # 建立StandardScaler对象
                zscore = preprocessing.StandardScaler()
                # 标准化处理
                Xs = zscore.fit_transform(Xs)
                Xt = zscore.fit_transform(Xt)

                bda = BDA(kernel_type='primal', dim=30, lamb=0.1, mu=1, mode='WBDA', gamma=1)
                acc, y_predict, list_acc = bda.fit_predict(Xs, Ys, Xt, Yt)
                print(i, j)
                print(acc)