function LogisticRegression()
    clc; %clear work command window
    clear variables;
    path="C:\Users\vxbha\OneDrive\Desktop\ML\Q4\";

    load("dataset4.mat");
    %disp(data.x);%disp for printing
    N = 0.1; %Step size
    E = 0.005; %Tolerance
    theta = rand(size(X,2),1);
    iterations_max=500000;
    limiter=0;
    % For plotting stats
    risk_values = [];
    error_values = [];
    theta_before = theta+4*E;
    while norm(theta - theta_before) >= E
        if limiter > iterations_max
            break;
        end
        
        r = risk (X, Y, theta);
        f = 1./(1+exp(-X*theta));
        f(f<0.5) = 0;
        f(f>=0.5) = 1;
        err = sum(f~=Y)/length(Y);
        fprintf('Iteration#:%d, Error Value:%0.5f,Risk Value:%0.5f\n', limiter, err,r);
        risk_values = cat(1,risk_values,r);
        error_values = cat(1, error_values, err);
        
        theta_before = theta;
        G = gradient(X,Y,theta);
        theta = theta - N*G;
        limiter = limiter + 1;
    end
    
    figure, plot (1:limiter, risk_values, 'r', 1: limiter, error_values, 'b');
    xlabel("# of iterations");
    ylabel("Error and risk");
    title("Risk and Error vs Iteration");
    
    legend("risk","error");
    
    plotoutput=sprintf("Plot for Stepsize %d and Tolerance %E.png",N,E);
    print(path+plotoutput,"-dpng");
    disp('Theta Values');
    disp(theta)
    x=0:0.01:1;
    y=(-theta(3) - theta(1).*x)/theta(2);
    figure, plot(x,y,"r"); 
    hold on;
    plot(X(:,1),X(:,2), '.');
    title('Linear Decision Boundary');
    plotoutput=("Linear Decision Boundary.png");
    print(path+plotoutput,"-dpng");
end



