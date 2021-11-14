%load the dataset
load ("dataset - Copy.mat"); %replaced 0's in original dataset file to -1's 

n= size (X,1);
sam_train = randsample (n, n/2);
sam_test = setdiff (1:n, sam_train);

trainX = X(sam_train, :);
testX = X(sam_test,:);
trainY = Y(sam_train,:);
testY = Y(sam_test,:);

global p1 %similar to that in svkernel.m;

c_vals1 = linspace(0.1,10);
err_linear = zeros(1,numel(c_vals1));

for idx=1:numel(c_vals1)
    C = c_vals1(idx);
    [nsv,alpha,bias] = svc(trainX, trainY,'linear', C);
    pred_Y = svcoutput(trainX,trainY,testX,'linear',alpha, bias);
    err_linear(idx)= svcerror(trainX,trainY,testX,testY,'linear',alpha,bias);
end

f = figure(1);
clf(f);
plot(c_vals1, err_linear);
xlabel('C');
ylabel('Error');
print(f, '-depsc', 'linear.eps');

c_vals2 = [0.00001,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,2,5,10,100];
degree_max = 14;
err_d = zeros(degree_max, numel(c_vals2));

for d=1:degree_max
    p1 = d; 
    for j=1:length(c_vals2)
        C=c_vals2(j);
        [nsv,alpha,bias] = svc(trainX, trainY,'poly', C);
        pred_Y = svcoutput(trainX,trainY,testX,'poly',alpha, bias);
        err_d(d,j)= svcerror(trainX,trainY,testX,testY,'poly',alpha,bias);
    end
end

f= figure(2);
clf(f);
var = [1:degree_max; log10(c_vals2); err_d].';
bar3(var);
xlabel('Degree of Polynomial');
ylabel('C (Log Scale)');
zlabel('Error');
print(f, '-depsc', 'poly.eps');

%rbfs
%keep sigma and cost on the log scale- dhananjai
sigmas = .1:.2:2;
c_vals3 = linspace(1,10,10);
err_sigma = zeros(numel(c_vals3),numel(sigmas));

for sigma_i=1:numel(sigmas)
    for j=1:length(c_vals3)
        C=c_vals3(j);
        p1 = sigmas(sigma_i);
        [nsv,alpha,bias] = svc(trainX, trainY,'rbf',C);
        pred_Y = svcoutput(trainX,trainY,testX,'rbf', alpha, bias);
        err_sigma(j,sigma_i)=svcerror(trainX,trainY,testX,testY,'rbf',alpha,bias);
    end
end

f=figure(3);
clf(f);
var = [log10(sigmas); c_vals3; err_sigma].';
bar3(var);
xlabel('Sigma (Log Scale)');
ylabel('C');
zlabel('Error');
print(f,'-depsc','rbf.eps');

disp('');
[~, idx] = min(err_linear);
fprintf('Linear Kernel: C = %f, Error = %f\n', c_vals1(idx), err_linear(idx));