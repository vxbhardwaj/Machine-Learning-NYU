function f = classification_func(theta,x,Y)
    y = x*theta;
    f = ones(length(x),1);
    for i=1:length(x)
        if y(i)*Y(i)<0
         f(i)=1;
        elseif y(i)*Y(i)>=0
            f(i)=0; %instead of -1, I have equated it to zero so the classification error converges to 0.
        end
    end
end