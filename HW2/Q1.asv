X= data(:,1:2);
Y= data(:,3);
stepsize = 1;
t=0;
theta=[0,0,1];
converged = 0;
iterations=0;
l=zeros(heigh(X),1);
e=zeros(heigh(X),1);
binary_error = zeros(10000,1);
perception_loss = zeros(10000,1);
while ~converged & iterations<10000
    converged = true;
    t=0;
    for i = 1:N 
        g= X(i,:)*theta';
        if(g>=0)
            g=1;
        else
            g=-1;
        end
        e(i)=abs(Y(i)-g);
        if Y(i)*g<=0
            t=t+1;
            converged=false;
            theta=theta+Y(i)*X(i,:);
            l(i,:) = stepsize*(-Y(i)*X(i,:)*theta');
        end
    end

    iterations = iterations+1
    binary_error(iterations) = sum(e)/N;
