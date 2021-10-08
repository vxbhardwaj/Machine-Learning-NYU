clc; %clear work command window
clear variables;
path="C:\Users\vxbha\OneDrive\Desktop\ML\Q2\";

data=load("problem2.mat");
%disp(data.x);%disp for printing

x=data.x;
y=data.y;

model_training_error=[];
model_testing_error=[];
model_parameters={};

i_indices=crossvalind("KFold",length(x),2);
x_train=x(i_indices==1,:);
y_train=y(i_indices==1);

x_test=x(i_indices==2,:);
y_test=y(i_indices==2);

lambdas=0:2000;

for lambda=lambdas
    [err,model,errT]=polyreg2(x_train,y_train,lambda,x_test,y_test);
    model_training_error(lambda+1)=err;
    model_testing_error(lambda+1)=errT;
    model_parameters{lambda+1}=model;
end

close all;
hold on;

plot(lambdas, model_testing_error,"b");
plot(lambdas, model_training_error,"g");

[~,index_variable]=min(model_testing_error);
plot(lambdas(index_variable),model_testing_error(index_variable),"b*");
xlabel("Lambda");
ylabel("Error");
legend("test","train");
title("Training and Testing Errors With Lambda");
plotoutput=("Error of training and testing for lambda 1 to 2000.png");
print(path+plotoutput,"-dpng");