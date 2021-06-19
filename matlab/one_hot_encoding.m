function [Y_onehot] = one_hot_encoding(Y, C)
    % intput:
    % Y: n*1, label vector
    % C: number of source class, C=length(unique(Ys))
    % output：
    % Y_onehot： n*C, one-hot label matrix
    % 使用one hot encoding 对labels进行编码，在计算数据的MMD矩阵时可以大大减少运算代价
    
    % 样本数量
    n = length(Y);
    Y_onehot = zeros(n, C);
    for i = 1:n
        class = Y(i);
        Y_onehot(i, class) = 1;
    end
end