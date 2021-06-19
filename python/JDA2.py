# encoding = utf-8

# quanxueliang 2020.3.8

import numpy as np 
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from numpy.matlib import repmat


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class JDA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1, T=10):
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.T = T
        print(self.lamb)

    def fit_predict(self, Xs, Ys, Xt, Yt):
        list_acc = []  # 存放每一轮迭代后的正确率
        X = np.hstack((Xs.T, Xt.T))  # X = [Xs.T, Xt.T]
        X = X / np.linalg.norm(X, axis=0)  # 归一化 
        m, n = X.shape  # 矩阵大小，m代表特征维数，n代表源域和目标域样本总数
        ns, nt = len(Xs), len(Xt)  
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        C = len(np.unique(Ys))
        H = np.eye(n) - 1 / n * np.ones((n, n))

        M = 0
        Y_tar_pseudo = None
        for t in range(self.T):
            N = 0
            M0 = e * e.T * C
            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
                for c in range(1, C+1):                     
                    e = np.zeros((n, 1))  # n代表源域和目标域样本总数 
                    Ns = len(Ys[np.where(Ys == c)]) # 计算源域属于类别c的样本数
                    Nt = len(Y_tar_pseudo[np.where(Y_tar_pseudo ==c)])  # 计算目标域伪标签为c的样本数
                    #print('Ns = {}, Nt = {}'.format(Ns, Nt))
                    '''
                    if Ns == 0:
                        Ns = 1
                    if Nt == 0:
                        Nt = 1   '''

                    tt = Ys == c  
                    e[np.where(tt == True)] = 1 / Ns
                    yy = Y_tar_pseudo == c  
                    ind = np.where(yy == True)  
                    inds = [item + ns for item in ind]  
                    e[tuple(inds)] = -1 / Nt
                                        
                    # np.isinf(ndarray) 返回一个判断是否是无穷的bool型数组
                    e[np.isinf(e)] = 0  # 执行本条语句后矩阵e元素并没有发生任何改变
                    # Mc 的维数是(ns+nt) * (ns+nt), 和矩阵 M0 的大小相同
                    N = N + np.dot(e, e.T)
            M = M0 + N # N=M1+M2+···+Mc
            M = M / np.linalg.norm(M, 'fro')  # 矩阵归一化

            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            # 计算矩阵(I + lam * KMK)-1 * KHK 的特征值与特征向量
            n_eye = m if self.kernel_type == 'primal' else n
            a = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye)
            b = np.linalg.multi_dot([K, H, K.T])
            # 计算特征值与特征向量
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)  # 将特征值小到大排序并返回序号
            # 有dim 个特征向量构成变换矩阵A
            A = V[:, ind[:self.dim]]
            # 计算变换后的核矩阵
            Z = np.dot(A.T, K)
            Z = Z / np.linalg.norm(Z, axis=0)  # 矩阵 列向量 范数
            Xs_new = Z[:, :ns].T
            Xt_new = Z[:, ns:].T 

            # 分类器
            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(Xs_new, Ys.ravel())  # 训练
            Y_tar_pseudo = clf.predict(Xt_new)  # 分类
            acc = sklearn.metrics.accuracy_score(Yt, Y_tar_pseudo)
            list_acc.append(acc) # 将参数acc添加到列表末尾
            print('JDA iteration [{}/{}]: Acc: {:.4f}'.format(t+1, self.T, acc))
        return acc, Y_tar_pseudo, list_acc



if __name__ == '__main__':
    # office_caltech10_surf dataset
    domains_surf = ['amazon_SURF_L10.mat', 'webcam_SURF_L10.mat', 'dslr_SURF_L10.mat', 'Caltech10_SURF_L10.mat']
    # office_caltech10_surf_zscore dataset
    domains_surf_zscore = ['amazon_zscore_SURF_L10.mat', 'webcam_zscore_SURF_L10.mat', 
                            'dslr_zscore_SURF_L10.mat', 'Caltech10_zscore_SURF_L10.mat']
    # MNIST_USPS_SURF dataset
    domains_MU = ['MNIST_SURF.mat', 'USPS_SURF.mat']
    # COIL20_SURF dataset
    domains_COIL20 = ['COIL1_SURF.mat', 'COIL2_SURF.mat']
    # PIE_SURF dataset
    domains_pie = ['pie05_surf', 'pie07_surf', 'pie09_surf', 'pie27_surf', 'pie29_surf']

    # i = 3
    # j = 2
    for i in range(2):
        for j in range(2):
            if i != j:
                # office-caltech10_surf dataset       
                '''
                src = 'F:\\MachineLearning\\Code\\data\\office_caltech_10_surf\\' + domains_surf[i]
                tar = 'F:\\MachineLearning\\Code\\data\\office_caltech_10_surf\\' + domains_surf[j]  '''
                # office_caltech10_surf_zscore dataset
                '''
                src = 'F:\\MachineLearning\\Code\\data\\office_caltech_10_surf_zscore\\' + domains_surf_zscore[i]
                tar = 'F:\\MachineLearning\\Code\\data\\office_caltech_10_surf_zscore\\' + domains_surf_zscore[j] '''
                # MNIST_USPS_SURF dataset        
                '''
                src = 'F:\\MachineLearning\\Code\\data\\USPS_MNIST_surf\\' + domains_MU[i]
                tar = 'F:\\MachineLearning\\Code\\data\\USPS_MNIST_surf\\' + domains_MU[j]   ''' 
                # COIL20_SURF dataset
                
                src = 'F:\MachineLearning\Code\data\COIL20_surf\\' + domains_COIL20[i]
                tar = 'F:\MachineLearning\Code\data\COIL20_surf\\' + domains_COIL20[j]   
                # PIE_SURF dataset
                '''
                src = 'F:\MachineLearning\Code\data\PIE_surf\\' + domains_pie[i]
                tar = 'F:\MachineLearning\Code\data\PIE_surf\\' + domains_pie[j]   '''

                src_domain = scipy.io.loadmat(src)
                tar_domain = scipy.io.loadmat(tar)
                Xs = src_domain['fts']
                Ys = src_domain['labels']
                Xt = tar_domain['fts']
                Yt = tar_domain['labels']

                #Z-Score标准化
                
                # 使特征行和为1
                Xs = Xs / repmat(np.mat(Xs.sum(axis=1)).T, 1, Xs.shape[1]) 
                Xt = Xt / repmat(np.mat(Xt.sum(axis=1)).T, 1, Xt.shape[1]) 
                #建立StandardScaler对象
                zscore = preprocessing.StandardScaler()
                # 标准化处理
                Xs = zscore.fit_transform(Xs)
                Xt = zscore.fit_transform(Xt)   

                jda = JDA(kernel_type='linear', dim=30, lamb=0.1, gamma=1)
                acc, y_predict, list_acc = jda.fit_predict(Xs, Ys, Xt, Yt)
                print(i, j)
                print(acc)





