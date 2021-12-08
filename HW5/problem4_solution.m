clc; clear variables;
values_of_p = cell(4, 1);
values_of_p{1} = [0.1, 0.7; 0.8, 0.3];
values_of_p{2} = [0.5, 0.1; 0.1, 0.5];
values_of_p{3} = [0.1, 0.5; 0.5, 0.1];
values_of_p{4} = [0.9, 0.3; 0.1, 0.3];
marg = values_of_p;
n = size(marg,1);
seprt = ones(n-1,2);
%Left To Right
for i=1:n-1
   seprt(i,:) = sum(marg{i});
   marg{i+1} = marg{i+1}.*(seprt(i,:)'*[1,1]);
end
%Right to Left
for i=1:n-1
  seprts_old = seprt(n-i,:);
   seprt(n-i,:) = sum(marg{n-i+1},2)';
   marg{n-i} = marg{n-i}.*([1;1]*(seprt(n-i,:)./seprts_old));
end
%Normalizing
for i=1:n
 marg{i} = marg{i}/sum(sum(marg{i}));
end
disp('Marginal Values Calculated:');
for i=1:n
    val = marg{i};
    disp(val);
end