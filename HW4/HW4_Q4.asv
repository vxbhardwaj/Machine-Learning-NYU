load('teapots.mat')
teapot_data = teapotImages;
m = mean(teapot_data);
X = teapot_data - m;
C = cov(X);
[V, D] = eig(C);
[d, ind] = sort(diag(D),'descend');
d = d(1:3,:);
v = V(:,ind(1:3));
c = X*v;
X_hat = m+c*v';

%10 images 
for i = 11:20
    figure(i);
    colormap gray;
    subplot(1,2,1);
    imagesc(reshape(teapot_data(i,:),38,50));
    title('Before Recon');
    axis image;
    subplot(1,2,2)
    imagesc(reshape(X_hat(i,:),38,50));
    title('After Recon');
    axis image;
end
norm(teapot_data-X_hat)