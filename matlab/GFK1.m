function [G] = GFK1(X_src,Y_src,X_tar,Y_tar,dim)
% Inputs:
%%% dim    :   dimension of geodesic flow kernel, dim <= 0.5 * n_feature
% Outputs:
%%% acc    :   accuracy after GFK and 1NN
%%% G      :   geodesic flow kernel matrix
%%% Cls    :   prediction labels for target, nt * 1

    Ps = pca(X_src);  % ʹ��matlab�Դ���pca()������pca���صľ���ÿһ�ж����໥������
    Pt = pca(X_tar);
    G = GFK_core([Ps,null(Ps')], Pt(:,1:dim));
    % �������ж�Ӧ��Q = [Ps��Rs]������ Rs = null(Ps')
    % Z = null(A) �Ǵ�����ֵ�ֽ��õ� A ����ռ��������
    % ����A����ռ�N(A)��ָ����Ax=0�����н�ļ���
    % [Ps,null(Ps')] = Ps
    % Pt(:,1:dim) = Pt��ǰdim��
%     [Cls, acc] = my_kernel_knn(G, X_src, Y_src, X_tar, Y_tar);
end


function [prediction,accuracy] = my_kernel_knn(M, Xr, Yr, Xt, Yt)
    dist = repmat(diag(Xr*M*Xr'),1,length(Yt)) ...
        + repmat(diag(Xt*M*Xt')',length(Yr),1)...
        - 2*Xr*M*Xt';
    [~, minIDX] = min(dist);
    prediction = Yr(minIDX);
    accuracy = sum(prediction==Yt) / length(Yt); 
end

function G = GFK_core(Q,Pt)
    % Input: Q = [Ps, null(Ps')], where Ps is the source subspace, column-wise orthonormal
    %        Pt: target subsapce, column-wise orthonormal, D-by-d, d < 0.5*D
    % Output: G = \int_{0}^1 \Phi(t)\Phi(t)' dt

    N = size(Q,2); % Q������
    dim = size(Pt,2); % Pt����������PCA�������ά��

    % compute the principal angles
    QPt = Q' * Pt;
    [V1,V2,V,Gam,Sig] = gsvd(QPt(1:dim,:), QPt(dim+1:end,:));
    % [U,V,X,C,S] = gsvd(A,B) �����Ͼ��� U �� V����ͨ�������� X �Լ��Ǹ��ԽǾ��� C �� S��
    % ��ʹA = U*C*X' B = V*S*X' C'*C + S'*S = I�� C��S�ǶԽ���������ֵ����
    V2 = -V2;
    theta = real(acos(diag(Gam))); % theta is real in theory. Imaginary part is due to the computation issue.

    % compute the geodesic flow kernel
    eps = 1e-20;
    B1 = 0.5.*diag(1+sin(2*theta)./2./max(theta,eps));
    B2 = 0.5.*diag((-1+cos(2*theta))./2./max(theta,eps));
    B3 = B2;
    B4 = 0.5.*diag(1-sin(2*theta)./2./max(theta,eps));
    G = Q * [V1, zeros(dim,N-dim); zeros(N-dim,dim), V2] ...
        * [B1,B2,zeros(dim,N-2*dim);B3,B4,zeros(dim,N-2*dim);zeros(N-2*dim,N)]...
        * [V1, zeros(dim,N-dim); zeros(N-dim,dim), V2]' * Q';
end


