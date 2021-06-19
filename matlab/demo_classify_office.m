% JPDA & JGSA 组合尝试

clc; clear all;

srcStr = {'amazon','amazon','amazon','webcam','webcam','webcam','dslr',  'dslr',  'dslr',   'caltech','caltech','caltech'};
tarStr = {'webcam','dslr', 'caltech','amazon','dslr', 'caltech','amazon','webcam','caltech', 'amazon','webcam', 'dslr'};


accuracy_list = [];
k = 0;
T = 10;
for i = 1:12
    src = char(srcStr{i});
    tar = char(tarStr{i});
    
    %% load source domian dataset
      load(['./data/' src '_SURF.mat']); 
%    fts = double(fts); 
    Xs = fts;
    Ys = labels;
    clear fts, clear labels;

    %% load target domain dataset
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
%     Xs = normr(Xs);                           % 行单位化
%     Xt = normr(Xt);
%     Xs = normc(Xs'); % Xs = m维 * n个样本
%     Xt = normc(Xt');
    Xs = Xs';  
    Xt = Xt';

    % JPDA evaluation
    options.p = 30;
    options.lambda = 0.1;
    options.ker = 'primal';
    options.mu = 0.1;
    options.gamma = 1.0;
    Cls = []; Acc = [];
    for t = 1:T
        [Zs,Zt] = JPDA0(Xs,Xt,Ys,Cls,options);
        mdl = fitcknn(Zs',Ys);
        Cls = predict(mdl,Zt');
        acc = length(find(Cls==Yt))/length(Yt);
%         Acc = [Acc;acc];
    end
    accuracy_list = [accuracy_list,acc];
%     fprintf('JPDA=%0.4f\n\n',accuracy_list(end));
    fprintf('%d: %s -> %s: Acc = %.4f\n',i,src,tar,acc);
end