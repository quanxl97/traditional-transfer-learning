%% QuanXueliang 
%% traditional_TR_test.m 的初始版本，后面有一些改进

clear all
%% load dataset
% datapath = './data/';
% office_caltech10_surf dataset
% office1 = 'F:\MachineLearning\Code\data\office_caltech_10_surf\amazon_SURF_L10.mat';
% office2 = 'F:\MachineLearning\Code\data\office_caltech_10_surf\webcam_SURF_L10.mat';
% office3 = 'F:\MachineLearning\Code\data\office_caltech_10_surf\dslr_SURF_L10.mat'; 
% office4 = 'F:\MachineLearning\Code\data\office_caltech_10_surf\Caltech10_SURF_L10.mat';
% domains_office_caltech10_surf = char(office1,office2,office3,office4);
srcStr = {'amazon','amazon','amazon','webcam','webcam','webcam','dslr',  'dslr',  'dslr',   'caltech','caltech','caltech',};
tarStr = {'webcam','dslr', 'caltech','amazon','dslr', 'caltech','amazon','webcam','caltech', 'amazon','webcam', 'dslr',};

% office_caltech10_surf_zscore dataset
% off_cal1 = 'F:\MachineLearning\Code\data\office_caltech_10_surf_zscore\amazon_zscore_SURF_L10.mat';  
% off_cal2 = 'F:\MachineLearning\Code\data\office_caltech_10_surf_zscore\webcam_zscore_SURF_L10.mat'; 
% off_cal3 = 'F:\MachineLearning\Code\data\office_caltech_10_surf_zscore\dslr_zscore_SURF_L10.mat'; 
% off_cal4 = 'F:\MachineLearning\Code\data\office_caltech_10_surf_zscore\Caltech10_zscore_SURF_L10.mat';
% domains_oaaice_caltch10_zscore = char(off_cal1,off_cal2,off_cal3,off_cal4);
% 
% 
% % MNIST_USPS_SURF dataset
% MNIST = 'F:\MachineLearning\Code\data\USPS_MNIST_surf\MNIST_SURF.mat';
% USPS = 'F:\MachineLearning\Code\data\USPS_MNIST_surf\USPS_SURF.mat';
% domains_MU = char(MNIST,USPS);
% 
% % COIL20_SURF dataset
% COIL1 = 'F:\MachineLearning\Code\data\COIL20_surf\COIL1_SURF.mat';
% COIL2 = 'F:\MachineLearning\Code\data\COIL20_surf\COIL2_SURF.mat';
% domains_COIL = char(COIL1,COIL2);
% 
% % PIE_SURF dataset
% PIE05 = 'F:\MachineLearning\Code\data\PIE_surf\pie05_surf.mat';
% PIE07 = 'F:\MachineLearning\Code\data\PIE_surf\pie07_surf.mat';
% PIE09 = 'F:\MachineLearning\Code\data\PIE_surf\pie09_surf.mat';
% PIE27 = 'F:\MachineLearning\Code\data\PIE_surf\pie27_surf.mat';
% PIE29 = 'F:\MachineLearning\Code\data\PIE_surf\pie29_surf.mat';
% domains_PIE = char(PIE05,PIE07,PIE09,PIE27,PIE29);
% 
% % office31_decaf6 dataset
% amazon_fc6 = 'F:\MachineLearning\Code\data\office31_decaf\amazon_fc6.mat';
% webcam_fc6 = 'F:\MachineLearning\Code\data\office31_decaf\webcam_fc6.mat';
% dslr_fc6 = 'F:\MachineLearning\Code\data\office31_decaf\dslr_fc6.mat';
% domains_office31_fc6 = char(amazon_fc6,webcam_fc6,dslr_fc6);
% 
% % office31_decaf7 dataset
% amazon_fc7 = 'F:\MachineLearning\Code\data\office31_decaf\amazon_fc7.mat';
% webcam_fc7 = 'F:\MachineLearning\Code\data\office31_decaf\webcam_fc7.mat';
% dslr_fc7 = 'F:\MachineLearning\Code\data\office31_decaf\dslr_fc7.mat';
% domains_office31_fc7 = char(amazon_fc7,webcam_fc7,dslr_fc7);


%% TL算法测试 ===================================================
  options.kernel_type = 'primal' ;           
  options.dim         = 30;                   
  options.lambda      = 0.1;                  
  options.gamma       = 1;                    
  options.T           = 10;
% % %   options.mode        = 'BDA';
% %   options.mu          = 0.5;

% JGSA parameters
% options.k = 100;            % #subspace bases 
% options.ker = 'primal';     % kernel type, default='linear' options: linear, primal, gauss, poly
% options.T = 10;             % #iterations, default=10
% options.alpha= 1;           % the parameter for subspace divergence ||A-B||
% options.mu = 1;             % the parameter for target variance
% options.beta = 0.1;        % the parameter for P and Q (source discriminaiton) 
% options.gamma = 2; 

% results = [];
accuracy_list = [];
k = 0;
for i = 1 : 12
%     for j = 1 : 4
%         if eq(i,j) == 0
            k = k+1;
            src = char(srcStr{i});
            tar = char(tarStr{i});
            
            %% load source domian dataset
%             load(domains_office_caltech10_surf(i,:));
              load(['./data/' src '_SURF.mat']);
%             load(domains_oaaice_caltch10_zscore(i,:));
%             load(domains_MU(i,:));
%             load(domains_COIL(i,:));
%             load(domains_PIE(i,:));
%             load(domains_office31_fc6(i,:));
%             load(domains_office31_fc7(i,:));           
%             fts = double(fts); 
            Xs = fts;
            Ys = labels;
            clear fts
            clear labels
            
            %% load target domain dataset
%             load(domains_office_caltech10_surf(j,:));
            load(['./data/' tar '_SURF.mat']);
%             load(domains_oaaice_caltch10_zscore(j,:));
%             load(domains_MU(j,:));
%             load(domains_COIL(j,:));
%             load(domains_PIE(j,:));
%             load(domains_office31_fc6(j,:));
%             load(domains_office31_fc7(j,:));          
%             fts = double(fts); 
            Xt = fts;
            Yt = labels;
            clear fts
            clear labels
            
            Xs = Xs./repmat(sum(Xs,2),1,size(Xs,2));  % 特征行和为1
            Xs = zscore(Xs,1);                        % 标准化，列和为0
            Xt = Xt./repmat(sum(Xt,2),1,size(Xt,2));
            Xt = zscore(Xt,1);
%             Xs = normr(Xs);                           % 行单位化
%             Xt = normr(Xt);
%             Xs = normc(Xs'); % Xs = m维 * n个样本
%             Xt = normc(Xt');
%             Xs = Xs';  
%             Xt = Xt';
%             
            
            % TCA 
%             [acc] = TCA(Xs, Ys, Xt, Yt, options);

            % JDA 
            [acc,acc_ite] = JDA(Xs, Ys, Xt, Yt, options);
            
            % JDA2
%             [acc] = JDA2(Xs, Xt, Ys, Yt, options);

            % BDA 
            % [acc,acc_ite,A] = BDA(Xs, Ys, Xt, Yt, options);
            
            % JGSA
%             [acc] = JGSA(Xs, Xt, Ys, Yt, options);

            % JGSA2
%             [acc] = JGSA2(Xs, Xt, Ys, Yt, options);

            % GFK 
%             [acc,G,Cls] = GFK(Xs, Ys, Xt, Yt, 30);

            % CORAL 
%             [acc,X_src_new] = CORAL(Xs, Ys, Xt, Yt);
            
%             fprintf('%d -> %d: Acc = %.4f\n',i,j,acc);
            fprintf('%d: %s -> %s: Acc = %.4f\n',i,src,tar,acc);
            accuracy_list(k) = acc;
%         end
%     end
end


 % construct MMD matrix
 % 计算类内,类间联合概率分布差异的MMD
   if ~isempty(Yt0) && length(Yt0)==nt
       %%% Ms
       Ms = 0; % e*e'*C; % 
       for c = reshape(unique(Ys),1,C)
           e = zeros(n,1);
           e(Ys==c) = 1 / ns;
           e(ns+find(Yt0==c)) = -1 / nt;
           e(isinf(e)) = 0;
           Ms = Ms + e*e';
       end
       %%% Md
       Md = 0; % e*e'*C; %
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
   Ms = X*Ms*X';
   Md = X*Md*X';




