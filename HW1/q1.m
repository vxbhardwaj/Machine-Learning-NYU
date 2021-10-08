clc; %clear work command window
clear variables;
path="C:\Users\vxbha\OneDrive\Desktop\ML\Q1\";

data=load("problem1.mat");
%disp(data.x);%disp for printing

x=data.x;
y=data.y;

model_training_error=[];
model_testing_error=[];
model_parameters={};

for d=1:8
    [err,model]=polyreg(x,y,d);
    model_training_error(d)=err;
    model_parameters{d}=model;
    title(sprintf("Polynomial of degree %d", d));
    plotoutput=sprintf("Plot output for degree %d.png",d);
    print(path+plotoutput,"-djpeg");
end

[~,index_variable]=min(model_training_error);
minimum_error=model_training_error(index_variable);

i_indices=crossvalind("KFold",length(x),2);
x_train=x(i_indices==1);
y_train=y(i_indices==1);

x_test=x(i_indices==2);
y_test=y(i_indices==2);

for d=1:60
    [err,model,errT]=polyreg(x_train,y_train,d,x_test,y_test);
    model_training_error(d)=err;
    model_testing_error(d)=errT;
    model_parameters{d}=model;
end

clf ; %clf is for remove the plot, but keep the axes
hold on ; %hold the last figure that has been plotted

plot(model_training_error,"r");
plot(model_testing_error,"b");
[~,index_variable]=min(model_testing_error);
plot(index_variable,model_testing_error(index_variable),"b*");
xlabel("strength of a polynomial");
ylabel("Error");
legend("train","test");
title("Training and Testing Errors-After cross-validation");
plotoutput=("Error of training and testing for d from 1 to 60.png");
print(path+plotoutput,"-dpng");