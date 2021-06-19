# encoding = utf-8

import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors
import os
from numpy.matlib import repmat
from sklearn import preprocessing

class CORAL:
    def __init__(self):
        super(CORAL, self).__init__()

    def fit(self, Xs, Xt):
        '''
        Perform CORAL on the source domain features
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: New source domain features
        '''
        cov_src = np.cov(Xs.T) + np.eye(Xs.shape[1])
        cov_tar = np.cov(Xt.T) + np.eye(Xt.shape[1])
        A_coral = np.dot(scipy.linalg.fractional_matrix_power(cov_src, -0.5),
                         scipy.linalg.fractional_matrix_power(cov_tar, 0.5))
        Xs_new = np.dot(Xs, A_coral)
        return Xs_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Perform CORAL, then predict using 1NN classifier
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted labels of target domain
        '''
        Xs_new = self.fit(Xs, Xt)
        clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        return acc, y_pred


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
    # office31_decaf6 dataset
    domains_office31_fc6 = ['\\amazon_fc6.mat', '\\webcam_fc6.mat', '\\dslr_fc6.mat']
    # office31_decaf7 dataset
    domains_office31_fc7 = ['\\amazon_fc7.mat', '\\webcam_fc7.mat', '\\dslr_fc7.mat']

    for i in range(2):
        for j in range(2):
            if i != j:
                print("sourct --> target: %d -> %d" % (i + 1, j + 1))
                datapath = os.getcwd()  # 获取当前路径，路径不包含文件名

                src = datapath + '\\data' + domains_pie[i]
                tar = datapath + '\\data' + domains_pie[j]
                # load data 数据集中每行表示一个样本
                src_domain = scipy.io.loadmat(src)
                tar_domain = scipy.io.loadmat(tar)
                Xs = src_domain['fts']
                Ys = src_domain['labels']
                Xt = tar_domain['fts']
                Yt = tar_domain['labels']

                '''
                X, Y = np.vstack((Xs,Xt)), np.vstack((Ys,Yt))
                #Ys, Yt, Y = Ys.ravel(), Yt.ravel(), Y.ravel()
                ns,m = Xs.shape
                nt,m = Xt.shape
                n,m = X.shape
                print(ns,nt,m)
                print(Ys.shape,Yt.shape,Y.shape)
                '''

                # Z-Score标准化
                '''
                # 使特征行和为1
                Xs = Xs / repmat(np.mat(Xs.sum(axis=1)).T, 1, Xs.shape[1])
                Xt = Xt / repmat(np.mat(Xt.sum(axis=1)).T, 1, Xt.shape[1])
                # 建立StandardScaler对象
                zscore = preprocessing.StandardScaler()
                # 标准化处理
                Xs = zscore.fit_transform(Xs)
                Xt = zscore.fit_transform(Xt)   '''

                '''
                # t分布-随机近邻嵌入 tSNE 
                print('Computing t-SNE embedding......')
                tsne = TSNE(n_components=3, init='pca', random_state=0)
                #Xs_tsne = tsne.fit_transform(Xs)
                #Xt_tsne = tsne.fit_transform(Xt)
                X_tsne = tsne.fit_transform(X)
                Xs_tsne_new = X_tsne[0:ns, :]
                Xt_tsne_new = X_tsne[ns:ns+nt, :]
                #m_tsne = Xs_tsne.shape[1]
                print(Xs_tsne_new.shape,Xt_tsne_new.shape)
                print(Xs_tsne_new.shape,Ys.shape)
                print("X_tsne done") 
                Xs = Xs_tsne_new
                Xt = Xt_tsne_new  '''

                coral = CORAL()
                acc, ypre = coral.fit_predict(Xs, Ys, Xt, Yt)
                print(acc)

                