function [acc] = JDA2(Xs, Xt, Ys, Yt, options)
% 目标：对源域和目标域分别找映射矩阵A,B
    %% Set options
    kernel_type = options.kernel_type;
    dim         = options.dim;
    T           = options.T;
    lambda      = options.lambda;  % 正则化项系数
    
    Xs = Xs';  % 转换成 m维 * n个样本
    Xt = Xt';
       
    % 开始迭代之前打上伪标签
    knn_model = fitcknn(Xs',Ys,'NumNeighbors',1);
    Yt0 = knn_model.predict(Xt');
    acc = length(find(Yt0==Yt))/length(Yt); 
    fprintf('acc=%0.4f\n',acc);
       
    m  = size(Xs,1);   % 特征维数
    ns = size(Xs,2);   % 源域样本数
    nt = size(Xt,2);

    class = unique(Ys); % unique()数据去重
    C = length(class);  % 类别数
    if strcmp(kernel_type,'primal')
        for t = 1:T  % 迭代，目的寻找更好的变换矩阵A,B
            % Construct MMD matrix
            [Ms, Mt, Mst, Mts] = constructMMD(ns,nt,Ys,Yt0,C);

            Ts = Xs*Ms*Xs';
            Tt = Xt*Mt*Xt';
            Tst = Xs*Mst*Xt';
            Tts = Xt*Mts*Xs';

            % Construct centering matrix
%             Hs = eye(ns)-1/(ns)*ones(ns,ns);
            Ht = eye(nt)-1/(nt)*ones(nt,nt);

            X = [zeros(m,ns) zeros(m,nt); zeros(m,ns) Xt];    
            H = [zeros(ns,ns) zeros(ns,nt); zeros(nt,ns) Ht];
            % St = Xt*H*Xt'

            Smax = X*H*X';
            Smin = [Ts+lambda*eye(m), Tst-lambda*eye(m);...
                    Tts-lambda*eye(m),  Tt+lambda*eye(m)];
            
            [W,~] = eigs(Smin+1e-9*eye(2*m), Smax, dim, 'SM');  
            % d = eigs(A,B,___) 解算广义特征值问题 A*V = B*V*D。
            A = W(1:m, :);
            B = W(m+1:end, :);

            Zs = A'*Xs;
            Zt = B'*Xt;

            if T>1
                %  更新伪标签
                knn_model = fitcknn(Zs',Ys,'NumNeighbors',1);
                Yt0 = knn_model.predict(Zt');
                acc = length(find(Yt0==Yt))/length(Yt); 
                fprintf('acc of iter %d: %0.4f\n',t, acc);
            end
        end
    end
end
