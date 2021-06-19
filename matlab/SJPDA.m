function [acc] = SJPDA(Xs,Xt,Ys,Yt,options)
% input Xs: n * m
% 函数说明：类别之间的概率差异计算方式采用JPDA，同时考虑类内、类间散度问题
% 参考JGSA，类内、类间散度考虑源域中的数据
kernel_type = options.kernel_type;
dim         = options.dim;
T           = options.T;
gamma       = options.gamma;
lambda      = options.lambda;
mu          = options.mu;

class = unique(Ys); % unique()数据去重
C = length(class);  % 类别数

% 计算源域数据的类内、类间散度
Xss = Xs';  % 转换成 m维 * n个样本
Xss = Xss*diag(sparse(1./sqrt(sum(Xss.^2))));  % normalization
m = size(Xss,1);
meanTotal = mean(Xss,2);  % mean(A,2)返回矩阵A每一行的均值
Sw = zeros(m, m);  % 用于存放类内scatter matrix
Sb = zeros(m, m);  % 用于存放类间scatter matrix
for i=1:C
    Xi = Xss(:,find(Ys==class(i)));
    meanClass = mean(Xi,2);
    Hi = eye(size(Xi,2))-1/(size(Xi,2))*ones(size(Xi,2),size(Xi,2));
    Sw = Sw + Xi*Hi*Xi'; % calculate within-class scatter
    Sb = Sb + size(Xi,2)*(meanClass-meanTotal)*(meanClass-meanTotal)'; % calculate between-class scatter
end
% 这里是否需要标准化是个问题！！JGSA中没有进行标准化处理
% Sw = Sw / norm(Sw,'fro');
% Sb = Sb / norm(Sb,'fro');

% 开始迭代之前打上伪标签
knn_model = fitcknn(Xs,Ys,'NumNeighbors',1);
Yt0 = knn_model.predict(Xt);
acc = length(find(Yt0==Yt))/length(Yt);
fprintf('acc=%0.4f\n',acc);

acc_ite = [];
%% iteration
for i = 1 : T
    [Z,A] = SJPDA_core(Xs,Xt,Ys,Yt0,Sw,Sb,options);
    % normalization
    Z = Z*diag(sparse(1./sqrt(sum(Z.^2)))); % k * n
    Zs = Z(:,1:size(Xs,1));
    Zt = Z(:,size(Xs,1)+1:end);
    
    knn_model = fitcknn(Zs',Ys,'NumNeighbors',1);
    Yt0 = knn_model.predict(Zt');
    acc = length(find(Yt0==Yt))/length(Yt);
    fprintf('JPDA+NN [%d/%d]=%.4f\n',i,T,acc);
    acc_ite = [acc_ite;acc];
end
end

function [Z,A] = SJPDA_core(Xs,Xt,Ys,Yt0,Sw,Sb,options)
kernel_type = options.kernel_type;
dim         = options.dim;
gamma       = options.gamma;
lambda      = options.lambda;
mu          = options.mu;

X = [Xs', Xt'];
X = X*diag(sparse(1./sqrt(sum(X.^2))));  % normalization
[m, n] = size(X);
ns = size(Xs,1);
nt = size(Xt,1);
e = [1/ns*ones(ns,1); -1/nt*ones(nt,1)];
C = length(unique(Ys));

% one hot encoding labels
[Ys_onehot] = one_hot_encoding(Ys,C);
[Yt0_onehot] = one_hot_encoding(Yt0,C);

% construct MMD matrix
if ~isempty(Yt0) && length(Yt0)==nt
    %%% Ms
    Ms = 0; % e*e'*C; %
    Ns = Ys_onehot/ns;
    Nt = Yt0_onehot/nt;
    Ms = [Ns*Ns', -Ns*Nt';  -Nt*Ns', Nt*Nt'];
    %%% Md
    Md = 0; % e*e'*C; %
    for i = 1:C
        Mc = repmat(Ys_onehot(:,i),1,C-1)/ns;
        Mt = [Yt0_onehot(:,1:i-1),Yt0_onehot(:,i+1:end)]/nt;
        Md = Md + [Mc*Mc', -Mc*Mt';  -Mt*Mc', Mt*Mt'];
    end
end
Ms = Ms / norm(Ms,'fro');
Md = Md / norm(Md,'fro');
Ms = X*Ms*X';
Md = X*Md*X';

%% center matrix H
H = eye(n) - 1/n * ones(n,n);

%% caculation
if strcmp(kernel_type,'primal')
    %        [A,~] = eigs(Ms+mu*Sw,lambda*Md,dim,'SM');
    % Ms, mu*Md, lambda*Sw, lambda*Sb, lambda*eye(m), X*H*X';
    % 1e-*eye(m)
    Smin = Ms+0.001*Sw+0.001*eye(m);  %-mu*Md+lambda*eye(m); % 
    Smax = 0.001*Md+0.001*Sb;  %X*H*X';  % 
    [A,~] = eigs(Smin,Smax,dim,'SM');
    A = real(A);
    %        [A,~] = eigs(Ms+0.01*Sw-mu*Md+lambda*eye(m),X*H*X',dim,'SM');
    Z = A'*X;
else
    K = kernel_jpdc(kernel_type,X,[],gamma);
    [A,~] = eigs(K*(Ms-mu*Md)*K'+lambda*eye(m),K*H*K',dim,'SM');
    Z = A'*K;
end
end


function K = kernel_jpdc(ker,X,X2,gamma)

switch ker
    case 'linear'
        
        if isempty(X2)
            K = X'*X;
        else
            K = X'*X2;
        end
        
    case 'rbf'
        
        n1sq = sum(X.^2,1);
        n1 = size(X,2);
        
        if isempty(X2)
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
        else
            n2sq = sum(X2.^2,1);
            n2 = size(X2,2);
            D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
        end
        K = exp(-gamma*D);
        
    case 'sam'
        
        if isempty(X2)
            D = X'*X;
        else
            D = X'*X2;
        end
        K = exp(-gamma*acos(D).^2);
        
    otherwise
        error(['Unsupported kernel ' ker])
end
end




