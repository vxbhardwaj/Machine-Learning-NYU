clc; %clear work command window
clear variables;
clf;
path="C:\Users\vxbha\Desktop\VB2182_ML_HW2\";

load("data3.mat");

%Initialization of Data
N=length(data);
X=[data(:,1:2),ones(N,1)]; %need to add array of all ones to accomodate bias

Y=data(:,3);
theta=randn(3,1); %3-by-1 vector of random numbers

missed_values=1;
%number_of_iterations=100;
current_iteration=1;
binary_classification_error=[];
perceptron_error=[];
%Employing SGD
while(missed_values~=0) % && current_iteration<=number_of_iterations) not converged, fewer than 10k iterations
    for i=1:N
        Loss=Y(i)*(X(i,:)*theta);
        if Loss<=0
            %Update
            theta = theta+Y(i)*X(i,:)';
            y1=classification_func(theta,X,Y);
            y2 = perceptron_func(theta,X,Y);
            binary_classification_error(current_iteration)=mean(y1);          
            perceptron_error(current_iteration)=mean(y2);
        end
    end
    missed_values=mean(y2);
    
    current_iteration=current_iteration+1;
end

figure(1)
for i=1:N
    if Y(i)==1
        plot(X(i,1),X(i,2),'.','MarkerSize',20,'MarkerEdgeColor','g');
    elseif Y(i)==-1
        plot(X(i,1),X(i,2),'.','MarkerSize',20,'MarkerEdgeColor','r');
    end
    hold on
end
xlabel('X1');
ylabel('X2');
x = [min(X(:,1)):1/200:max(X(:,1))];
x2 = (-theta(3)*ones(N,1)-theta(1)*x)/theta(2);
plot(x,x2);
title("Linear Decision Boundary on 2d x data");
plotoutput=("Linear Decision Boundary on 2d x data.png");
print(path+plotoutput,"-dpng");

figure(2)
plot(binary_classification_error);
xlabel('# of Iterations');
ylabel('Binary Classification Error');
title("Binary Classification Error vs # of Iterations");
plotoutput=("Binary Classification Error vs # of Iterations.png");
print(path+plotoutput,"-dpng");

figure(3)
plot(perceptron_error);
xlabel('# of Iterations');
ylabel('Perceptron Error');
title("Perceptron Error vs # of Iterations");
plotoutput=("Perceptron Error vs # of Iterations.png");
print(path+plotoutput,"-dpng");