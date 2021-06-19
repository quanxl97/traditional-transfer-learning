# encoding = utf-8

import numpy as np
from sklearn.cluster import KMeans
#import matplotlib
#matplotlib.use('TkAgg')  # 出图
import matplotlib.pyplot as plt
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from numpy.matlib import repmat


#from dataset import dataset
from test import test
# import test # 好像不能直接调用



def get_data(real_center= [(1, 1), (1, 2), (2, 2), (2, 1)]):
	# 先在四个中心点附近产生一堆数据
	point_number = 50

	points_x = []
	points_y = []

	for center in real_center:
		offset_x, offset_y = np.random.randn(point_number) * 0.3, np.random.randn(point_number) * 0.25
		x_val, y_val = center[0] + offset_x, center[1] + offset_y
		points_x.append(x_val)
		points_y.append(y_val)

	points_x = np.concatenate(points_x)
	points_y = np.concatenate(points_y)
	p_list = np.stack([points_x, points_y], axis=1)
	return p_list

# # real_center = [(1, 1), (1, 2), (2, 2), (2, 1)]
# # points=get_data(real_center)


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
    # office31_decaf6 dataset
    domains_office31_fc6 = ['amazon_fc6.mat', 'webcam_fc6.mat', 'dslr_fc6.mat']
    # office31_decaf7 dataset
    domains_office31_fc7 = ['amazon_fc7.mat', 'webcam_fc7.mat', 'dslr_fc7.mat']
    
    
    i = 1
    j = 2
    
    data = get_data()
    kmeans_model = KMeans(n_clusters=4, n_init=15).fit(data)
    # n_init: 获取初始簇中心的更迭次数
    labels = kmeans_model.labels_   
    print(labels)
    print(data)
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow')
    plt.show()
    print("ok")
    test()

    # office-caltech10_surf dataset       
    '''
    src = 'F:\\MachineLearning\\Code\\data\\office_caltech_10_surf\\' + domains_surf[i]
    tar = 'F:\\MachineLearning\\Code\\data\\office_caltech_10_surf\\' + domains_surf[j] '''
    # office_caltech10_surf_zscore dataset
    
    src = 'F:\\MachineLearning\\Code\\data\\office_caltech_10_surf_zscore\\' + domains_surf_zscore[i]
    tar = 'F:\\MachineLearning\\Code\\data\\office_caltech_10_surf_zscore\\' + domains_surf_zscore[j]  
    # MNIST_USPS_SURF dataset  
    '''
    src = 'F:\\MachineLearning\\Code\\data\\USPS_MNIST_surf\\' + domains_MU[i]
    tar = 'F:\\MachineLearning\\Code\\data\\USPS_MNIST_surf\\' + domains_MU[j]  '''
    # COIL20_SURF dataset
    '''
    src = 'F:\MachineLearning\Code\data\COIL20_surf\\' + domains_COIL20[i]
    tar = 'F:\MachineLearning\Code\data\COIL20_surf\\' + domains_COIL20[j]  '''
    # PIE_SURF dataset
    '''
    src = 'F:\MachineLearning\Code\data\PIE_surf\\' + domains_pie[i]
    tar = 'F:\MachineLearning\Code\data\PIE_surf\\' + domains_pie[j]   '''
    # office31_decaf6 dataset
    '''
    src = 'F:\MachineLearning\Code\data\office31_decaf\\' + domains_office31_fc6[i]
    tar = 'F:\MachineLearning\Code\data\office31_decaf\\' + domains_office31_fc6[j]   '''
    # office31_decaf7 dataset
    '''
    src = 'F:\MachineLearning\Code\data\office31_decaf\\' + domains_office31_fc7[i]
    tar = 'F:\MachineLearning\Code\data\office31_decaf\\' + domains_office31_fc7[j]   '''

    src_domain = scipy.io.loadmat(src)
    tar_domain = scipy.io.loadmat(tar)
    Xs = src_domain['fts']
    Ys = src_domain['labels']
    Xt = tar_domain['fts']
    Yt = tar_domain['labels']

    kmeans_model = KMeans(n_clusters=10, n_init=15).fit(Xt)
    # n_init: 获取初始簇中心的更迭次数
    labels = kmeans_model.labels_  
    print(labels) 
    plt.scatter(Xt[:, 0], Xt[:, 1], c=labels, cmap='rainbow')
    plt.show()
    print("xt ok")

