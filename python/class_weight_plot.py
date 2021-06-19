# encoding = utf-8

import numpy as np 
from numpy.matplot import repmat


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
    

    


