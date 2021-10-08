function Remp = risk (x,y, theta)
    f = 1./(1+exp(-x*theta));
    r = (y-1).*log(1-f)-y.*log(f);
    r(isnan(r))=0; %if not a number
    Remp=mean(r); %Summation divided by N
end