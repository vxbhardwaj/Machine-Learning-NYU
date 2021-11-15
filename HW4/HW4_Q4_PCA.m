load('teapots.mat')
X = teapotImages;
[coefficient_of_3, score3] = pca(X,'Algorithm','eig','Rows','all','NumComponents',3);
Xhat3 = mean(X)+score3*coefficient_of_3';
[coefficient_of_6, score6] = pca(X,'Algorithm','eig','Rows','all','NumComponents',6);
Xhat6 = mean(X)+score6*coefficient_of_6';
[coefficient_of_32, score32] = pca(X,'Algorithm','eig','Rows','all','NumComponents',32);
Xhat32 = mean(X)+score32*coefficient_of_32';


figure(1);
colormap gray;
subplot(2,2,1);
imagesc(reshape(data(10,:),38,50));
title('Before');
axis image;
subplot(2,2,2)
imagesc(reshape(Xhat3(10,:),38,50));
title('TOP3');
axis image;
subplot(2,2,3)
imagesc(reshape(Xhat6(10,:),38,50));
title('TOP6');
axis image;
subplot(2,2,4)
imagesc(reshape(Xhat32(10,:),38,50));
title('TOP32');
axis image;
