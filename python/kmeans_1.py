# coding='utf-8'
# quan xueliang
"""t-SNE对数据进行可视化"""

from time import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn import datasets
from sklearn.manifold import TSNE
import scipy.linalg
import sklearn.metrics

from sklearn.cluster import KMeans
from numpy.matlib import repmat
from sklearn import preprocessing
 
 
def get_data():
    digits = datasets.load_digits(n_class=10)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features
 
 
def plot_embedding(data, label, title):
    # 归一化
#    x_min, x_max = np.min(data, 0), np.max(data, 0)
#    data = (data - x_min) / (x_max - x_min)
 
    fig = plt.figure()
    plt.xlim(-80,80)
    plt.ylim(-80,80)
    # ax = plt.subplot(111)
    # 绘制散点图
    plt.scatter(data[:,0], data[:,1], s=0, c=label, cmap='rainbow')
    # 标记标签
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.))  # 标签颜色设置
#                 fontdict={'weight': 'bold', 'size': 9})  # 标签文本字体设置
#    plt.xticks([])
#    plt.yticks([])
    plt.title(title)
    return fig

def plot_scatter(data,label,title):
#    x_min, x_max = np.min(data, 0), np.max(data, 0)
#    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    plt.xlim(-80,80)
    plt.ylim(-80,80)
    plt.scatter(data[:,0], data[:,1], c=label, cmap='rainbow')
    plt.title(title)
    return fig

 
 
def main():
    domains_MU = ['MNIST_SURF.mat', 'USPS_SURF.mat']
    i = 0
    j = 1
    src = 'F:\\MachineLearning\\Code\\data\\USPS_MNIST_surf\\' + domains_MU[i]
    tar = 'F:\\MachineLearning\\Code\\data\\USPS_MNIST_surf\\' + domains_MU[j]
    src_domain = scipy.io.loadmat(src)
    tar_domain = scipy.io.loadmat(tar)
    Xs,Ys = src_domain['fts'],src_domain['labels']
    Xt,Yt = tar_domain['fts'],tar_domain['labels']

    X, Y = np.vstack((Xs,Xt)), np.vstack((Ys,Yt))
    Ys, Yt, Y = Ys.ravel(), Yt.ravel(), Y.ravel()
    ns,m = Xs.shape
    nt,m = Xt.shape
    n,m = X.shape
    print(ns,nt,m)
    print(Ys.shape,Yt.shape,Y.shape)
    
    # tSNE 降维
    print('Computing t-SNE embedding......')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    Xs_tsne = tsne.fit_transform(Xs)
    Xt_tsne = tsne.fit_transform(Xt)
    X_tsne = tsne.fit_transform(X)
    Xs_tsne_new = X_tsne[0:ns, :]
    Xt_tsne_new = X_tsne[ns:ns+nt, :]
    m_tsne = Xs_tsne.shape[1]
    print(Xs_tsne_new.shape,Xt_tsne_new.shape)
    print(Xs_tsne_new.shape,Ys.shape)
    print("X_tsne done") 
    
    # 绘制带标签的数据图
    print("draw fig with labels")
    fig = plot_embedding(Xs_tsne_new, Ys, 'MNIST with labels after tSNE embedding of MNIST+USPS')
    plt.show(fig)
    print("draw fig with labels ---------- 1")

    fig = plot_embedding(Xt_tsne_new, Yt, 'USPS with labels after tSNE embedding of MNIST+USPS')
    plt.show(fig)

    # 对降维后的数据聚类，并进行绘图
    print("kmeans cluster after tsne")
    kmeans_model = KMeans(n_clusters=10, n_init=10).fit(Xs_tsne_new)
    Ys0 = kmeans_model.labels_
    print("draw fig after cluster without labels")
    fig = plot_scatter(Xs_tsne_new, Ys0, 'KMeans of MNIST after tSNE of MNIST+USPS')
    plt.show(fig)
    kmeans_model = KMeans(n_clusters=10, n_init=10).fit(Xt_tsne_new)
    Yt0 = kmeans_model.labels_
    print("draw fig after cluster without labels")
    fig = plot_scatter(Xt_tsne_new, Yt0, 'KMeans of USPS after tSNE of MNIST+USPS')
    plt.show(fig)
    print("kmeans cluster done ---------- 2")

    # 对原始数据进行聚类，降维后进行可视化
    print("kmeans cluster wuthout tsne")
    kmeans_model_no_tsne = KMeans(n_clusters=10, n_init=10).fit(X)
    Y0 = kmeans_model_no_tsne.labels_
    fig = plot_scatter(X_tsne, Y0, 'KMeans of MNIST+USPS without tSNE')
    plt.show(fig)
    print("kmeans cluster without tsne done ---------- 3")
    
    # 类别标签处理
    # Ys_class = list(set(Ys))
    Ys_class = np.unique(Ys)
    C = len(Ys_class)
    print(Ys_class,C)
    Xs_c = 0
    # Xs_tsne_c = 0
    center_c = np.empty((C,m_tsne))
    # 把所有标签为c的样本提取出来
    for c in range (1,C+1):
        Ns_tsne_c = len(Ys[np.where(Ys == c)])
        Ys_c = [c] * Ns_tsne_c
        Xs_tsne_c = np.empty((Ns_tsne_c,m_tsne))
        Xs_tsne_c_mean = 0
#        distence = np.empty((Ns_tsne_c,1))
#        distence = list(distence)
        j = 0
        for i in range(ns):
            if Ys[i] == c:
                Xs_tsne_c[j,:] = Xs_tsne[i,:]
                j = j + 1
#        for i in range(Ns_tsne_c):
#            for j in range(Ns_tsne_c):
#                distence[i] = distence[i] + np.sqrt(np.sum((Xs_tsne_c[i,:]-Xs_tsne_c[j,:])**2))
#        ind = distence.index(min(distence))
        Xs_tsne_c_mean = np.mean(Xs_tsne_c,axis=0)
        center_c[c-1,:] = Xs_tsne_c_mean  #Xs_tsne_c[ind,:]
        fig = plot_embedding(Xs_tsne_c, Ys_c, 'class of MNIST with labels after tSNE embedding of MNIST+USPS')
        plt.show(fig)
    print(center_c)
    # 把每个类别样本质心的图画出来
    fig = plot_embedding(center_c, Ys_class, 'class center of MNIST with labels after tSNE embedding of MNIST+USPS')
    plt.show(fig)

    print("main()...done")

if __name__ == '__main__':
    main()