# coding='utf-8'

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

 
def main():
    domains_MU = ['MNIST_SURF.mat', 'USPS_SURF.mat']
    i = 0
    j = 1
    
  #  src = 'F:\\MachineLearning\\Code\\data\\office_caltech_10_surf\\' + domains_surf[i]
    src = 'F:\\MachineLearning\\Code\\data\\USPS_MNIST_surf\\' + domains_MU[i]
    tar = 'F:\\MachineLearning\\Code\\data\\USPS_MNIST_surf\\' + domains_MU[j]
    src_domain = scipy.io.loadmat(src)
    tar_domain = scipy.io.loadmat(tar)
    Xs,Ys = src_domain['fts'],src_domain['labels']
    Xt,Yt = tar_domain['fts'],tar_domain['labels']
    X, Y = np.vstack((Xs,Xt)), np.vstack((Ys,Yt))
    Ys,Yt, Y = Ys.ravel(), Yt.ravel(), Y.ravel()
    ns,m = Xs.shape
    nt,m = Xt.shape
    n,m = X.shape

    print(ns,nt,m)
    print(Ys.shape,Yt.shape,Y.shape)

    ns = len(Xs)
    nt = len(Xt)
    print(ns,nt)
    

    ids = [1,1,2,2,3,3,4,4,5,4,4,3,2]
    ids1 = list(set(ids))
    print(ids1)
    print(ids1[1])
    
    plt.text(12,23,str(00))
    plt.show()

    a = np.array([[1,2],[3,4]])
    b = np.array([[7,8],[6,9]])
    c = b[0,:]-a[0,:]
    d = np.sum(c**2)
    e = np.mean(b,axis=0)
    f = [2] * 4
    print(a,b,c,d,e,f)


if __name__ == '__main__':
    main()