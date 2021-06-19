%% QuanXueliang  2020.3.17
clc; clear all;

%% load dataset
% datapath = './data/';
% office_caltech10_surf dataset
% srcStr = {'amazon','amazon','amazon','webcam','webcam','webcam','dslr',  'dslr',  'dslr',   'caltech','caltech','caltech'};
% tarStr = {'webcam','dslr', 'caltech','amazon','dslr', 'caltech','amazon','webcam','caltech', 'amazon','webcam', 'dslr'};

% % MNIST_USPS_SURF dataset
srcStr = {'MNIST','USPS'};
tarStr = {'USPS','MNIST'};
% %  
% % COIL20_SURF dataset
% srcStr = {'COIL1','COIL2'};
% tarStr = {'COIL2','COIL1'};

% % PIE_SURF dataset
% srcStr = {'PIE05','PIE05','PIE05','PIE05',  'PIE07','PIE07','PIE07','PIE07',  'PIE09','PIE09','PIE09','PIE09',...
%           'PIE27','PIE27','PIE27','PIE27',  'PIE29','PIE29','PIE29','PIE29'};
% tarStr = {'PIE07','PIE09','PIE27','PIE29',  'PIE05','PIE09','PIE27','PIE29',  'PIE05','PIE07','PIE27','PIE29',...
%           'PIE05','PIE07','PIE09','PIE29',  'PIE05','PIE07','PIE09','PIE27'};
%      
%% TL算法测试
  options.kernel_type = 'primal' ;           
  options.dim         = 100;                   
  options.lambda      = 0.1;                  
  options.gamma       = 1;                    
  options.T           = 10;
  options.mu          = 0.1;
% % %   options.mode        = 'BDA';


% JGSA parameters
% options.k = 30;            % #subspace bases 
% options.ker = 'primal';     % kernel type, default='linear' options: linear, primal, gauss, poly
% options.T = 10;             % #iterations, default=10
% options.alpha= 1;           % the parameter for subspace divergence ||A-B||
% options.mu = 1;             % the parameter for target variance
% options.beta = 0.1;        % the parameter for P and Q (source discriminaiton) 
% options.gamma = 2; 

accuracy_list = [];
for i = 1
    src = char(srcStr{i});
    tar = char(tarStr{i});

    % load source domian dataset
      load(['./data/' src '_SURF.mat']); 
%    fts = double(fts); 
    Xs = fts;
    Ys = labels;
    clear fts, clear labels;

    % load target domain dataset
    load(['./data/' tar '_SURF.mat']);      
%    fts = double(fts); 
    Xt = fts;
    Yt = labels;
    clear fts; clear labels;

    % data preprocessing
    Xs = Xs./repmat(sum(Xs,2),1,size(Xs,2));  % 特征行和为1
    Xs = zscore(Xs,1);                        % 标准化，列和为0
    Xt = Xt./repmat(sum(Xt,2),1,size(Xt,2));
    Xt = zscore(Xt,1);
% Xs=Xs';
% Xt=Xt';
%     Xs = Xs*diag(sparse(1./sqrt(sum(Xs.^2))));
%     Xt = Xt*diag(sparse(1./sqrt(sum(Xt.^2))));
%     Xs=Xs';
% Xt=Xt';
%     Xs = normr(Xs);                           % 行单位化
%     Xt = normr(Xt);
%     Xs = Xs';  
%     Xt = Xt';

%     [acc] = TCA(Xs, Ys, Xt, Yt, options);            % TCA 
%     [acc,acc_ite] = JDA(Xs, Ys, Xt, Yt, options);    % JDA 
%     [acc] = JDA2(Xs, Xt, Ys, Yt, options);           % JDA2
%     [acc,acc_ite,A] = BDA(Xs, Ys, Xt, Yt, options);  % BDA
%     [acc] = JPDA(Xs,Xt,Ys,Yt,options)                % JPDA
%     [acc] = WJPDA(Xs,Xt,Ys,Yt,options)                % WJPDA
%     [acc] = SJPDA(Xs,Xt,Ys,Yt,options)                % SJPDA
%     [acc] = JGSA(Xs, Xt, Ys, Yt, options);           % JGSA
%     [acc] = JGSA2(Xs, Xt, Ys, Yt, options);          % JGSA2
    [acc,G,Cls] = GFK(Xs, Ys, Xt, Yt, 20);           % GFK 
%     [acc,X_src_new] = CORAL(Xs, Ys, Xt, Yt);         % CORAL

    fprintf('%d: %s -> %s: Acc = %.4f\n',i,src,tar,acc);
    accuracy_list = [accuracy_list,acc];
end




