function [acc,X_src_new] = CORAL(X_src,Y_src,X_tar,Y_tar)
% quanxueliang  2020.3.21

	cov_src = cov(X_src) + eye(size(X_src,2));
	cov_tar = cov(X_tar) + eye(size(X_tar,2));
	A_coral = cov_src^(-1/2) * cov_tar^(1/2);
	X_src_new = X_src * A_coral;
    
    % ues knn to predict the target label
    knn_model = fitcknn(X_src_new, Y_src, 'NumNeighbors', 1);
    Y_tar_pseudo = knn_model.predict(X_tar);   
    acc = length(find(Y_tar_pseudo==Y_tar))/length(Y_tar);
end