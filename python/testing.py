
import scipy.io
import os

def main():
    domains_MU = ['\\MNIST_SURF', '\\USPS_SURF']
    # 获取当前路径，路径不包含文件名
    print(os.getcwd())
    src = os.getcwd() + '\\data' + domains_MU[1]
    source_domain = scipy.io.loadmat(src)
    Xs,Ys = source_domain['fts'], source_domain['labels']
    print('load done')
    print('ok')
    a = 100
    b = 5
    c = a/b
    d = a//b
    print(c,d)

    print('ok')


if __name__ == '__main__':
    main()