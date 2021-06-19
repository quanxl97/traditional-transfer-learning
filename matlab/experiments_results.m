%% 实验结果
clc; clear all
close all

%% MNIST-USPS  ==============================================
% task:M-U, U-M ，没有数据预处理
% TCA  primal 30 0.1
acc_TCA = [0.5878	0.5305];
% JDA primal	30	0.1
acc_JDA = [0.721111111	0.6];
% CORAL
acc_CORAL = [0.6561	0.469];
% GFK 20
acc_GFK = [0.643	0.445];
% JGSA nr	primal 	30	0.01
acc_JGSA = [0.7906	0.686];
% JPDA primal	30	0.1  0.1
acc_JPDA = [0.721111111	0.6];
% WJPDA  primal  30  0.1  0.01
acc_WJPDA = [0.761666667	0.6355];
% SJPDA  primal 30 
acc_SJPDA = [0.5006 0.4560];
task_mu = {'M-U','U-M'};
y_mu = [acc_TCA;
          acc_JDA;
          acc_JPDA;
          acc_WJPDA;
          acc_SJPDA]';
%% COIL20   ====================================================
% task:C1-C2, C2-C1
% TCA,1, zscore, primal, 30, 0.1
acc_TCA_coil = [0.8889	0.8639];
% JDA，1, zscore, primal, 30, 0.1
acc_JDA_coil = [0.9431	0.9444];
% CORAL 没有数据预处理
acc_CORAL_coil = [0.8569	0.8694];
% GFK,没有数据预处理，25
acc_GFK_coil = [0.8778	0.8806];
% JGSA，1, zscore, primal, 30, 0.1
acc_JGSA_coil = [0.954166667	0.976388889];
% JPDA，1, zscore, primal, 30, 0.1
acc_JPDA_coil = [0.934722222	0.888888889];
% WJPDA，1 zscore primal 30 0.1  0.001
acc_WJPDA_coil = [0.944444444	0.9375];
% SJPDA，考虑散度的JPDA, 1, zscore, 30, 0.001
acc_SJPDA_coil = [0.943055556	0.938888889];
task_coil20 = {'C1-C2','C2-C1'};
y_coil20 = [acc_TCA_coil;
            acc_JDA_coil;
            acc_JPDA_coil;
            acc_WJPDA_coil;
            acc_SJPDA_coil]';
%% Office-Caltech10-surf-dataset  ================================
% task:{'A-W','A-D','A-C', 'W-A','W-D','W-C', 'D-A','D-W','D-C', 'C-A','C-W','C-D'}
% TCA,1, zscore,   primal, 30, 0.1
acc_TCA_oc = [0.3831    0.3694    0.4007    0.3090    0.9172    0.3143    0.3309    0.8847    0.3161    0.4760    0.3424    0.4841];
% JDA，1, zscore,  primal, 30, 0.1
acc_JDA_oc = [0.4034    0.4076    0.3829    0.2923    0.8917    0.3206    0.3299    0.9119    0.3019    0.4562    0.4169    0.4522];
% CORAL, 1, zscore
acc_CORAL_oc = [0.2678    0.2675    0.2538    0.2620    0.8408    0.2262    0.2881    0.8441    0.3001    0.2359    0.2373    0.2611];
% GFK, 1,zscore,  30
acc_GFK_oc = [0.3797    0.3822    0.4025    0.3309    0.8854    0.2885    0.3361    0.8271    0.2858    0.4008    0.3559    0.4140];
% JGSA，1, zscore, primal, 30, 0.1
acc_JGSA_oc = [0.4576    0.4713    0.4150    0.3987    0.9045    0.3321    0.3800    0.9186    0.2992    0.5146    0.4542    0.4586];
% JPDA，1, zscore, primal, 30, 0.1
acc_JPDA_oc = [0.4576    0.4331    0.4025    0.3956    0.9045    0.3161    0.3706    0.8915    0.3117    0.4948    0.4237    0.4650];
% WJPDA，1 zs primal 30 0.1  0.001
acc_WJPDA_oc = [ 0.4068    0.4204    0.3891    0.4061    0.9045    0.2778    0.3100    0.8983    0.2983    0.4990    0.4441    0.4331];
% SJPDA，考虑散度的JPDA, 1, zscore, 30, 0.001
acc_SJPDA_oc = [0.4169    0.4204    0.4363    0.3121    0.8280    0.3259    0.4238    0.8407    0.3446    0.5418    0.4407    0.3885];
task_oc = {'A-W','A-D','A-C', 'W-A','W-D','W-C', 'D-A','D-W','D-C', 'C-A','C-W','C-D'};
y_oc = [acc_TCA_oc;  
        acc_JDA_oc;
        acc_JPDA_oc;
        acc_WJPDA_oc;
        acc_SJPDA_oc]';
    
%% 绘制条形图 ==================================================
x = task_mu;
y = y_mu; 
result = subplot(1,1,1);
hold(result,'on');

bar1 = bar(y);  % 条形图
% 创建 subplot
% mu_result = subplot(1,1,1,'Parent',Parent1);
hold(result,'on');
% 
% % 使用 bar 的矩阵输入创建多行
% % bar1 = bar(mu_result,'Parent',mu_result);
% bar1 = bar(y,'Parient',mu_result);
% set(bar1(1),'DisplayName','TCA');
% set(bar1(2),'DisplayName','JDA');
% set(bar1(3),'DisplayName','JPDA');
% set(bar1(4),'DisplayName','WJPDA');
% 插入图例
leg = legend('TCA','JDA','JPDA','WJPDA','SJPDA');

% 标题
title('Accuracy on MNIST+USPS');
% 坐标轴范围
% xlim(result,[0.5, 3.5]);
ylim(result,[0, 1.2]);
% 坐标轴标签
ylabel('Accuracy');
box(result,'on');
% 设置其余坐标属性
set(result,'XTick',[1:12],'XTickLabel',x);


















