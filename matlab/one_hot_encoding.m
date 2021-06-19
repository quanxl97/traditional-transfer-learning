function [Y_onehot] = one_hot_encoding(Y, C)
    % intput:
    % Y: n*1, label vector
    % C: number of source class, C=length(unique(Ys))
    % output��
    % Y_onehot�� n*C, one-hot label matrix
    % ʹ��one hot encoding ��labels���б��룬�ڼ������ݵ�MMD����ʱ���Դ������������
    
    % ��������
    n = length(Y);
    Y_onehot = zeros(n, C);
    for i = 1:n
        class = Y(i);
        Y_onehot(i, class) = 1;
    end
end