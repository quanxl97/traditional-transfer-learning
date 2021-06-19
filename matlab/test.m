% clear all

Nt=[1 2 3; 
   4 5 6];

idx=1:3; idx(1)=[];
 Mt=Nt(:,idx);


% 
% % aa = [-1,2,55,1;
% %     3,-4,67,2;
% %     50,60,0,36];
% % 
% % bb = normr(aa);
% % cc = normc(aa);
% 
% aa = [1+1i,1,2+3i,2,3,3];
% dd = real(aa);
% % ee = real(aa.);
% cc = length(unique(aa));
% % bb = reshape(unique(aa),1,3);
% 
% % 类间
% A= [1 1 0 0  0 -1 0 0 ]'*[1 1 0 0  0 -1 0 0 ];
% A = A + [1 1 0 0  0  0 -1 -1]'*[1 1 0 0  0  0 -1 -1];
% A = A + [0 0 1 0  -1 0  0  0 ]'*[0 0 1 0  -1 0  0  0 ];
% A = A + [0 0 1 0  0  0  -1 -1]'*[0 0 1 0  0  0  -1 -1];
% A = A + [0 0 0 1  -1 0  0  0]'*[0 0 0 1  -1 0  0  0];
% A = A + [0 0 0 1  0  -1 0  0]'*[0 0 0 1  0  -1 0  0];
% 
% % 类内
% C0 = [1 1 0 0  -1 0  0  0]'*[1 1 0 0  -1 0  0  0];
% C0 = C0+[0 0 1 0  0 -1 0 0  ]'*[0 0 1 0  0 -1 0 0  ];
% C0 = C0+[0 0 0 1  0  0  -1 -1 ]'*[0 0 0 1  0  0  -1 -1 ];
% 
% 
% Ys = [1 0 0;1 0 0 ;0 1 0 ;0 0 1];
% Yt = [1 0 0;0 1 0 ;0 0 1 ;0 0 1];
% % 类内
% D=[Ys*Ys', -Ys*Yt'; -Yt*Ys', Yt*Yt'];
% % Ys = Ys*2;
% ii=zeros(4,1);
% % 类间
% B=0;
% for i = 1:3
%     Fs = repmat(Ys(:,i),1,2);
%     Ft = [Yt(:,1:i-1),Yt(:,i+1:end)];
%     B=B+[Fs*Fs', -Fs*Ft'; -Ft*Fs', Ft*Ft'];
% end
% 
% Ys = [1;1;2;3];
% Yt = [1;2;3;3];
% ns = length(Ys);
% nt = length(Yt);
% C = length(unique(Ys));
% [Ys_onehot] = one_hot_encoding(Ys);
% [Yt_onehot] = one_hot_encoding(Yt);
% 
% Ms = 0;
% Ns = Ys_onehot;
% Nt = Yt_onehot;
% Ms = [Ns*Ns', -Ns*Nt';  -Nt*Ns', Nt*Nt'];
% 
% Md = 0; % 类间
% for i = 1 : C
%    Ns = repmat(Ys_onehot(:,i),1,C-1)/ns;
%    Nt = [Yt_onehot(:,1:i-1),Yt_onehot(:,i+1:end)]/ns;
%    Md = Md + [Ns*Ns', -Ns*Nt';  -Nt*Ns', Nt*Nt'];
% end
%    
% 
% 
% 
% 
% 
