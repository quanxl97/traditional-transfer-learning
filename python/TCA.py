# encoding = utf-8
# quan xueliang 2020.3.3
# 2020.7.2 检查过，没有问题

import numpy as np 
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from numpy.matlib import repmat
from sklearn import svm
from dataset import dataset
from sklearn import datasets
from sklearn.manifold import TSNE
import os


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal': 
        K = X1
    elif ker == 'linear':  
        if X2 is not None:  # if X2:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)   
    elif ker == 'rbf':  
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)   
    return K


class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb  # 正则化项系数
        self.gamma = gamma

    def fit(self, Xs, Xt):
        X = np.hstack((Xs.T, Xt.T))  # 水平方向堆叠数组
        X = X / np.linalg.norm(X, axis=0)  # 归一化
        m, n = X.shape  # 矩阵大小
        ns, nt = len(Xs), len(Xt) # 矩阵行数
        # 垂直方向堆叠数组
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        # 矩阵 L
        L = e * e.T
        # linalg = linear（线性）+ algebra（代数），norm则表示范数。
        L = L / np.linalg.norm(L, 'fro')
        # 中心矩阵
        # np.eye(n) 返回n*n的数组，主对角线元素为1，其余为0，n维的单位矩阵
        H = np.eye(n) - 1 / n * np.ones((n, n))
        # 核矩阵
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        # n_eye = m if self.kernel_type == 'primal' else n
        if self.kernel_type == 'primal':
            n_eye = m
        else:
            n_eye = n

        # 求矩阵(I + lam * KLK)-1 * KHK 的特征值与特征向量
        a = np.linalg.multi_dot([K, L, K.T]) + self.lamb * np.eye(n_eye)
        b = np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)  # 获取特征值与特征向量
        ind = np.argsort(w)  # 特征值小到大排序，返回索引序号
        # 获取特征变换矩阵
        A = V[:, ind[:self.dim]]  # 逆序切片后获取dim个特征向量
        # 得到新的核矩阵
        Z = np.dot(A.T, K)
        Z = Z / np.linalg.norm(Z, axis=0)  # 归一化
        Xs_new = Z[:, :ns].T
        Xt_new = Z[:, ns:].T
        return Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        Xs_new, Xt_new = self.fit(Xs, Xt)
        # 分类
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(Xs_new, Ys.ravel())  # 训练分类器
        y_pred = clf.predict(Xt_new)  # 预测标签
        # 计算准确率
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        print("tca_knn_acc: %f" % (acc))

        svm_clf = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr')
        svm_clf.fit(Xs_new, Ys.ravel())
        y_pred_svm = svm_clf.predict(Xt_new)
        acc_svm = sklearn.metrics.accuracy_score(Yt, y_pred_svm)
        print("tca_svm_acc: %f" %(acc_svm))
        return acc

def main():
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
                print("sourct --> target: %d -> %d" %(i+1, j+1))
                datapath = os.getcwd()  # 获取当前路径，路径不包含文件名
                # dataset: domains_surf, domains_surf_zscore, domains_MU, domains_COIL20
                # domains_pie, domains_office31_fc6, domains_office31_fc7
                # office-caltech10_surf dataset       
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

                # 使特征行和为1
                Xs = Xs / repmat(np.mat(Xs.sum(axis=1)).T, 1, Xs.shape[1])
                Xt = Xt / repmat(np.mat(Xt.sum(axis=1)).T, 1, Xt.shape[1])
                # 建立StandardScaler对象
                zscore = preprocessing.StandardScaler()
                # 标准化处理
                Xs = zscore.fit_transform(Xs)
                Xt = zscore.fit_transform(Xt)

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

                tca = TCA(kernel_type='primal', dim=30, lamb=0.1, gamma=1)
                acc = tca.fit_predict(Xs, Ys, Xt, Yt)
                print('ok')


if __name__ == '__main__':
    main()






        




