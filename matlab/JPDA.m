function [acc] = JPDA(Xs,Xt,Ys,Yt,options)

kernel_type = options.kernel_type;
dim         = options.dim;
T           = options.T;
gamma       = options.gamma;
lambda      = options.lambda;
mu          = options.mu;

% 开始迭代之前打上伪标签
%     knn_model = fitcknn(Xs,Ys,'NumNeighbors',1);
%     Yt0 = knn_model.predict(Xt);
%     acc = length(find(Yt0==Yt))/length(Yt);
%     fprintf('acc=%0.4f\n',acc);

% conputer related parameters
X = [Xs', Xt'];
X = X*diag(sparse(1./sqrt(sum(X.^2))));  % normalization
[m, n] = size(X);
ns = size(Xs,1);
nt = size(Xt,1);
e = [1/ns*ones(ns,1); -1/nt*ones(nt,1)];
C = length(unique(Ys));

Yt0 = [];
acc_ite = [];
% iteration
for t = 1 : T
    % 采用独热编码计算MMD矩阵
    % one hot encoding
    [Ys_onehot] = one_hot_encoding(Ys,C);
    Ns = Ys_onehot/ns;
    Nt = zeros(nt,C);
    
    % construct MMD matrix
    if ~isempty(Yt0) && length(Yt0)==nt
        [Yt0_onehot] = one_hot_encoding(Yt0,C);
        Nt = Yt0_onehot/nt;
    end
    %%% Ms
    Ms = [Ns*Ns', -Ns*Nt';  -Nt*Ns', Nt*Nt'];
    %%% Md
    Md = 0; % e*e'*C; %
    for i = 1:C
        Mc = repmat(Ns(:,i),1,C-1);
        Mt = [Nt(:,1:i-1),Nt(:,i+1:end)];
        %            idx=1:C; idx(i)=[];
        %            Mt=Nt(:,idx);
        Md = Md + [Mc*Mc', -Mc*Mt';  -Mt*Mc', Mt*Mt'];
    end
    Ms = Ms / norm(Ms,'fro');
    Md = Md / norm(Md,'fro');
    
    % center matrix H
    H = eye(n)-1/n*ones(n,n);

    % caculation
    if strcmp(kernel_type,'primal')
        [A,~] = eigs(X*(Ms-mu*Md)*X'+lambda*eye(m),X*H*X',dim,'SM');
        Z = A'*X;
    else
        K = kernel_jpda(kernel_type,X,[],gamma);
        [A,~] = eigs(K*(Ms-mu*Md)*K'+lambda*eye(n),K*H*K',dim,'SM');
        Z = A'*K;
    end
    
    % normalization
    Z = Z*diag(sparse(1./sqrt(sum(Z.^2)))); % k * n
    Zs = Z(:,1:size(Xs,1));
    Zt = Z(:,size(Xs,1)+1:end);
    
    knn_model = fitcknn(Zs',Ys,'NumNeighbors',1);
    Yt0 = knn_model.predict(Zt');
    acc = length(find(Yt0==Yt))/length(Yt);
    fprintf('JPDA+NN [%d/%d]=%.4f\n',t,T,acc);
    acc_ite = [acc_ite;acc];
end
end

function [Z,A] = JPDA_core(Xs,Xt,Ys,Yt0,options)
kernel_type = options.kernel_type;
dim         = options.dim;
gamma       = options.gamma;
lambda      = options.lambda;
mu          = options.mu;

% conputer related parameters
X = [Xs', Xt'];
X = X*diag(sparse(1./sqrt(sum(X.^2))));  % normalization
[m, n] = size(X);
ns = size(Xs,1);
nt = size(Xt,1);
e = [1/ns*ones(ns,1); -1/nt*ones(nt,1)];
C = length(unique(Ys));

% 采用独热编码计算MMD矩阵
% one hot encoding
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
        Mc = repmat(Ns(:,i),1,C-1);
        Mt = [Nt(:,1:i-1),Nt(:,i+1:end)];
        %            idx=1:C; idx(i)=[];
        %            Mt=Nt(:,idx);
        Md = Md + [Mc*Mc', -Mc*Mt';  -Mt*Mc', Mt*Mt'];
    end
    
    %        % 没有采用独热编码计算MMD矩阵
    %        %%% Ms
    %        Ms = 0;  % e*e'*C;  %
    %        for c = reshape(unique(Ys),1,C)
    %            e = zeros(n,1);
    %            e(Ys==c) = 1 / ns;
    %            e(ns+find(Yt0==c)) = -1 / nt;
    %            e(isinf(e)) = 0;
    %            Ms = Ms + e*e';
    %        end
    %
    %        %%% Md
    %        Md = 0;  % e*e'*C;  %
    %        for cs = reshape(unique(Ys),1,C)
    %            for ct = reshape(unique(Ys),1,C)
    %                if eq(cs,ct) == 0
    %                    e = zeros(n,1);
    %                    e(Ys==cs) = 1/ns;
    %                    e(ns+find(Yt0==ct)) = -1 / nt;
    %                    e(isinf(e)) = 0;
    %                    Md = Md + e*e';
    %                end
    %            end
    %        end
end
Ms = Ms / norm(Ms,'fro');
Md = Md / norm(Md,'fro');

% center matrix H
H = eye(n)-1/n*ones(n,n);

% caculation
if strcmp(kernel_type,'primal')
    [A,~] = eigs(X*(Ms-mu*Md)*X'+lambda*eye(m),X*H*X',dim,'SM');
    Z = A'*X;
else
    K = kernel_jpda(kernel_type,X,[],gamma);
    [A,~] = eigs(K*(Ms-mu*Md)*K'+lambda*eye(n),K*H*K',dim,'SM');
    Z = A'*K;
end
end

function K = kernel_jpda(ker,X,X2,gamma)

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


