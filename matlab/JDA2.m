function [acc] = JDA2(Xs, Xt, Ys, Yt, options)
% Ŀ�꣺��Դ���Ŀ����ֱ���ӳ�����A,B
    %% Set options
    kernel_type = options.kernel_type;
    dim         = options.dim;
    T           = options.T;
    lambda      = options.lambda;  % ������ϵ��
    
    Xs = Xs';  % ת���� mά * n������
    Xt = Xt';
       
    % ��ʼ����֮ǰ����α��ǩ
    knn_model = fitcknn(Xs',Ys,'NumNeighbors',1);
    Yt0 = knn_model.predict(Xt');
    acc = length(find(Yt0==Yt))/length(Yt); 
    fprintf('acc=%0.4f\n',acc);
       
    m  = size(Xs,1);   % ����ά��
    ns = size(Xs,2);   % Դ��������
    nt = size(Xt,2);

    class = unique(Ys); % unique()����ȥ��
    C = length(class);  % �����
    if strcmp(kernel_type,'primal')
        for t = 1:T  % ������Ŀ��Ѱ�Ҹ��õı任����A,B
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
            % d = eigs(A,B,___) �����������ֵ���� A*V = B*V*D��
            A = W(1:m, :);
            B = W(m+1:end, :);

            Zs = A'*Xs;
            Zt = B'*Xt;

            if T>1
                %  ����α��ǩ
                knn_model = fitcknn(Zs',Ys,'NumNeighbors',1);
                Yt0 = knn_model.predict(Zt');
                acc = length(find(Yt0==Yt))/length(Yt); 
                fprintf('acc of iter %d: %0.4f\n',t, acc);
            end
        end
    end
end