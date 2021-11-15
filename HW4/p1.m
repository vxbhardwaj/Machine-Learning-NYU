clear;clc;
load('teapots.mat');

%plot unaltered 10 images
figure(1);
for i=1:10
    subplot(4,5,i);
    imagesc(reshape(teapotImages(10*i,:),38,50));
    colormap gray;
    hold;
end

sigma=cov(teapotImages);
[eigenvector,eigenvalue]=eig(sigma);
a = sort(max(eigenvalue),'descend');
[~,b]=find(eigenvalue>a(4));

% plot of the 3 top eigenvectors
h=figure(2);
for i=1:3
    subplot(1,3,i);
    disp(eigenvector(:,b(i)))
    imagesc(reshape(eigenvector(:,b(i)),38,50));
    colormap gray;
    hold;
end

%plot of the mean
figure(3);    
    u = mean( teapotImages,1 );
    imagesc(reshape(u,38,50));
    colormap gray;
    hold;

for i=1:10
    reconstruct(:,i)= ( u'+i*eigenvector(:,b(1))+3.3*eigenvector(:,b(2))+(6.7-i)*eigenvector(:,b(3)) );
end

% show the reconstructed data
figure(1);
for i=1:10
    subplot(4,5,i+10);
    imagesc(reshape(reconstruct(:,i),38,50));
    colormap gray;
    hold;
end









