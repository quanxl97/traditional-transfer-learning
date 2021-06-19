function [acc] = JGSA(Xs, Xt, Ys, Yt, options)
% quanxueliang, input: n * m

    %% Set options
    ker         = options.ker;
    k           = options.k;
    T           = options.T;
    alpha       = options.alpha;
    mu          = options.mu;
    beta        = options.beta;
    gamma       = options.gamma;

    Xs = Xs';  % 转换成 m维 * n个样本
    Xt = Xt';
    
    % 开始迭代之前打上伪标签
    knn_model = fitcknn(Xs',Ys,'NumNeighbors',1);
    Yt0 = knn_model.predict(Xt');
    acc = length(find(Yt0==Yt))/length(Yt); 
    fprintf('acc=%0.4f\n',acc);
       
    m  = size(Xs,1);
    ns = size(Xs,2);
    nt = size(Xt,2);

    class = unique(Ys); % unique()数据去重
    C = length(class);  % 类别数
    if strcmp(ker,'primal')
    
        %---------------------------------------------------------------------
        % compute LDA
        dim = size(Xs,1);
        meanTotal = mean(Xs,2);  % mean(A,2)返回矩阵A每一行的均值

        Sw = zeros(dim, dim);  % 用于存放类内scatter matrix
        Sb = zeros(dim, dim);  % 用于存放类间scatter matrix
        for i=1:C
            Xi = Xs(:,find(Ys==class(i)));
            meanClass = mean(Xi,2);
            Hi = eye(size(Xi,2))-1/(size(Xi,2))*ones(size(Xi,2),size(Xi,2));
            Sw = Sw + Xi*Hi*Xi'; % calculate within-class scatter
            Sb = Sb + size(Xi,2)*(meanClass-meanTotal)*(meanClass-meanTotal)'; % calculate between-class scatter
        end
        P = zeros(2*m,2*m);
        P(1:m,1:m) = Sb;
        Q = Sw;

        for t = 1:T  % 迭代，目的寻找更好的变换矩阵A,B
            % Construct MMD matrix
            [Ms, Mt, Mst, Mts] = constructMMD(ns,nt,Ys,Yt0,C);

            Ts = Xs*Ms*Xs';
            Tt = Xt*Mt*Xt';
            Tst = Xs*Mst*Xt';
            Tts = Xt*Mts*Xs';

            % Construct centering matrix
            Ht = eye(nt)-1/(nt)*ones(nt,nt);

            X = [zeros(m,ns) zeros(m,nt); zeros(m,ns) Xt];    
            H = [zeros(ns,ns) zeros(ns,nt); zeros(nt,ns) Ht];
            % St = Xt*H*Xt'

            Smax = mu*X*H*X'+beta*P;
            Smin = [Ts+alpha*eye(m)+beta*Q, Tst-alpha*eye(m) ; ...
                    Tts-alpha*eye(m),  Tt+(alpha+mu)*eye(m)];
            [W,~] = eigs(Smax, Smin+1e-9*eye(2*m), k, 'LM');  
            % d = eigs(A,B,___) 解算广义特征值问题 A*V = B*V*D。
            A = W(1:m, :);
            B = W(m+1:end, :);

            Zs = A'*Xs;
            Zt = B'*Xt;

            if T>1
                %  更新伪标签
%                 Yt0 = knnclassify(Zt',Zs',Ys,1);  
                knn_model = fitcknn(Zs',Ys,'NumNeighbors',1);
                Yt0 = knn_model.predict(Zt');
                acc = length(find(Yt0==Yt))/length(Yt); 
                fprintf('acc of iter %d: %0.4f\n',t, full(acc));
            end
        end
    else
    
        % 使用核函数 kernel function
        Xst = [Xs, Xt];   
        nst = size(Xst,2); 
        [Ks, Kt, Kst] = constructKernel(Xs,Xt,ker,gamma);  % 计算核函数
       %--------------------------------------------------------------------------
        % compute LDA
        dim = size(Ks,2);
        C = length(class);
        meanTotal = mean(Ks,1);

        Sw = zeros(dim, dim);
        Sb = zeros(dim, dim);
        for i=1:C
            Xi = Ks(find(Ys==class(i)),:);
            meanClass = mean(Xi,1);
            Hi = eye(size(Xi,1))-1/(size(Xi,1))*ones(size(Xi,1),size(Xi,1));
            Sw = Sw + Xi'*Hi*Xi; % calculate within-class scatter
            Sb = Sb + size(Xi,1)*(meanClass-meanTotal)'*(meanClass-meanTotal); % calculate between-class scatter
        end
        P = zeros(2*nst,2*nst);
        P(1:nst,1:nst) = Sb;
        Q = Sw;        

        for t = 1:T

            % Construct MMD matrix
            [Ms, Mt, Mst, Mts] = constructMMD(ns,nt,Ys,Yt0,C);

            Ts = Ks'*Ms*Ks;
            Tt = Kt'*Mt*Kt;
            Tst = Ks'*Mst*Kt;
            Tts = Kt'*Mts*Ks;

            K = [zeros(ns,nst), zeros(ns,nst); zeros(nt,nst), Kt];
            Smax =  mu*K'*K+beta*P;

            Smin = [Ts+alpha*Kst+beta*Q, Tst-alpha*Kst;...
                    Tts-alpha*Kst, Tt+mu*Kst+alpha*Kst];
            [W,~] = eigs(Smax, Smin+1e-9*eye(2*nst), k, 'LM');
            W = real(W);

            A = W(1:nst, :);
            B = W(nst+1:end, :);

            Zs = A'*Ks';
            Zt = B'*Kt';

            if T>1
%                 Yt0 = knnclassify(Zt',Zs',Ys,1);  
                knn_model = fitcknn(Zs',Ys,'NumNeighbors',1);
                Yt0 = knn_model.predict(Zt');
                acc = length(find(Yt0==Yt))/length(Yt); 
                fprintf('acc of iter %d: %0.4f\n',t, full(acc));
            end
        end
end
