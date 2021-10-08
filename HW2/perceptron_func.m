function f = perceptron_func(theta,x,Y)
    y=x*theta;
    f=zeros(length(x),1);
    for i=1:length(x)
        if Y(i)*y(i)<0
            f(i)=-Y(i)*y(i);
        end
    end
end