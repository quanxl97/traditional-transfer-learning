function [acc,acc_ite] = JDA(X_src,Y_src,X_tar,Y_tar,options)
% Outputs:
%%% acc            :     final accuracy using knn, float
%%% acc_ite        :     list of all accuracies during iterations

% Set options
lambda      = options.lambda;
dim         = options.dim;
kernel_type = options.kernel_type;
gamma       = options.gamma;
T           = options.T;

acc_ite = [];
Y_tar_pseudo = [];
% Iteration
for i = 1 : T
    [Z,A] = JDA_core(X_src,Y_src,X_tar,Y_tar_pseudo,options);
    % normalization for better classification performance
    Z = Z*diag(sparse(1./sqrt(sum(Z.^2)))); % k * n
    Zs = Z(:,1:size(X_src,1));
    Zt = Z(:,size(X_src,1)+1:end);
    
    knn_model = fitcknn(Zs',Y_src,'NumNeighbors',1);
    Y_tar_pseudo = knn_model.predict(Zt');
    %         Y_tar_pseudo = knnclassify(Zt',Zs',Y_src,1);
    acc = length(find(Y_tar_pseudo==Y_tar))/length(Y_tar);
    fprintf('JDA+NN [%d/%d]=%0.4f\n',i,T,acc);
    acc_ite = [acc_ite;acc];
end
end

function [Z,A] = JDA_core(X_src,Y_src,X_tar,Y_tar_pseudo,options)
% Set options
lambda = options.lambda;              
dim = options.dim;                   
kernel_type = options.kernel_type;    
gamma = options.gamma;                

% Construct MMD matrix
X = [X_src', X_tar'];
X = X * diag(sparse(1./sqrt(sum(X.^2))));
[m, n] = size(X);
ns = size(X_src,1);
nt = size(X_tar,1);
e = [1/ns*ones(ns,1); -1/nt*ones(nt,1)];
C = length(unique(Y_src));


%%% M0
% ====这里要不要乘以C是个问题===============================?????????
M = e * e' * C;  % multiply C for better normalization

%%% Mc
N = 0;
if ~isempty(Y_tar_pseudo) && length(Y_tar_pseudo)==nt
    for c = reshape(unique(Y_src),1,C)
        e = zeros(n,1);
        e(Y_src==c) = 1 / length(find(Y_src==c));
        e(ns+find(Y_tar_pseudo==c)) = -1 / length(find(Y_tar_pseudo==c));
        e(isinf(e)) = 0;
        N = N + e*e';
    end
end

M = M + N;
M = M / norm(M,'fro');

% Centering matrix H
H = eye(n) - 1/n * ones(n,n);

% Calculation
if strcmp(kernel_type,'primal')
    [A,~] = eigs(X*M*X'+lambda*eye(m),X*H*X',dim,'SM');
    Z = A'*X;
else
    K = kernel_jda(kernel_type,X,[],gamma);
    [A,~] = eigs(K*M*K'+lambda*eye(n),K*H*K',dim,'SM');
    Z = A'*K;
end

end


function K = kernel_jda(ker,X,X2,gamma)

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

