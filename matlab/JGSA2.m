function [acc] = JGSA2(Xs, Xt, Ys, Yt, options)
% quanxueliang, input: n * m
% �ֲ�����ʹ�����ϸ��ʷֲ���ѧϰ����ӳ��ֱ�����Դ���Ŀ����
    %% Set options
    ker         = options.ker;
    k           = options.k;
    T           = options.T;
    alpha       = options.alpha;
    mu          = options.mu;
    beta        = options.beta;
    gamma       = options.gamma;

    Xs = Xs';  % ת���� mά * n������
    Xt = Xt';
       
    % ��ʼ����֮ǰ����α��ǩ
    knn_model = fitcknn(Xs',Ys,'NumNeighbors',1);
    Yt0 = knn_model.predict(Xt');
    acc = length(find(Yt0==Yt))/length(Yt); 
    fprintf('acc=%0.4f\n',acc);
       
    m  = size(Xs,1);
    ns = size(Xs,2);
    nt = size(Xt,2);
    n = ns + nt;

    class = unique(Ys); % unique()����ȥ��
    C = length(class);  % �����
    if strcmp(ker,'primal')
    
        %---------------------------------------------------------------------
        % compute LDA
        dim = size(Xs,1);
        meanTotal = mean(Xs,2);  % mean(A,2)���ؾ���Aÿһ�еľ�ֵ

        Sw = zeros(dim, dim);  % ���ڴ������scatter matrix
        Sb = zeros(dim, dim);  % ���ڴ�����scatter matrix
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

        for t = 1:T  % ������Ŀ��Ѱ�Ҹ��õı任����A,B
            % Construct MMD matrix
%             [Ms, Mt, Mst, Mts] = constructMMD(ns,nt,Ys,Yt0,C);
% 
%             Ts = Xs*Ms*Xs';
%             Tt = Xt*Mt*Xt';
%             Tst = Xs*Mst*Xt';
%             Tts = Xt*Mts*Xs';
            
                     % construct MMD matrix
           if ~isempty(Yt0) && length(Yt0)==nt
               %%% Ms
               Ms = 0;
               for c = reshape(unique(Ys),1,C)
                   e = zeros(n,1);
                   e(Ys==c) = 1 / ns;
                   e(ns+find(Yt0==c)) = -1 / nt;
                   e(isinf(e)) = 0;
                   Ms = Ms + e*e';
               end
               %%% Md
               Md = 0;
               for cs = reshape(unique(Ys),1,C)
                   for ct = reshape(unique(Ys),1,C)
                       if eq(cs,ct) == 0
                           e = zeros(n,1);
                           e(Ys==cs) = 1 / ns;
                           e(ns+find(Yt0==ct)) = -1 / nt;
                           e(isinf(e)) = 0;
                           Md = Md + e*e';
                       end
                   end
               end
           end
           Ms = Ms / norm(Ms,'fro');
           Md = Md / norm(Md,'fro');
              X = [Xs, Xt];
   X = X*diag(sparse(1./sqrt(sum(X.^2))));  % normalization
           Ms = X*Ms*X';
           Md = X*Md*X';
           

            % Construct centering matrix
%             Hs = eye(ns)-1/(ns)*ones(ns,ns);
            Ht = eye(nt)-1/(nt)*ones(nt,nt);

            X = [zeros(m,ns) zeros(m,nt); zeros(m,ns) Xt];    
            H = [zeros(ns,ns) zeros(ns,nt); zeros(nt,ns) Ht];
            % St = Xt*H*Xt'

            Smax = mu*X*H*X'+beta*P;
%             Smax = Smax + 0.1*Md;
%             Smin = [Ts+alpha*eye(m)+beta*Q, Tst-alpha*eye(m) ; ...
%                     Tts-alpha*eye(m),  Tt+(alpha+mu)*eye(m)];
            Smin = [alpha*eye(m)+beta*Q, alpha*eye(m) ; ...
                alpha*eye(m),  (alpha+mu)*eye(m)];
%             Smin = Smin + Ms; 
            [W,~] = eigs(Smax, Smin+1e-9*eye(2*m), k, 'LM');  
            % d = eigs(A,B,___) �����������ֵ���� A*V = B*V*D��
            A = W(1:m, :);
            B = W(m+1:end, :);

            Zs = A'*Xs;
            Zt = B'*Xt;

            if T>1
                %  ����α��ǩ
%                 Yt0 = knnclassify(Zt',Zs',Ys,1);  
                knn_model = fitcknn(Zs',Ys,'NumNeighbors',1);
                Yt0 = knn_model.predict(Zt');
                acc = length(find(Yt0==Yt))/length(Yt); 
                fprintf('acc of iter %d: %0.4f\n',t, full(acc));
            end
        end
    else
        
        
        
        
        
        
    
        % ʹ�ú˺��� kernel function
        Xst = [Xs, Xt];   
        nst = size(Xst,2); 
        [Ks, Kt, Kst] = constructKernel(Xs,Xt,ker,gamma);  % ����˺���
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