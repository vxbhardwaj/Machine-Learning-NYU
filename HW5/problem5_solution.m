clc; clear variables;
emis_p = [0.4, 0.1, 0.3, 0.2; 0.1, 0.4, 0.2, 0.3];
init = [1; 0];
obs_st = [1, 4, 2, 2, 3];
trans_p = [0.8, 0.2; 0.2, 0.8];
t_size = size(trans_p, 1);nsize = size(obs_st, 2);
val1 = zeros(t_size, t_size, nsize);val2 = zeros(t_size, nsize);
val2(:, 1) = init;
% L to R
for i = 2:nsize
   val = obs_st(1, i);
   val1(:, :, i) = diag(val2(:, i - 1)) *trans_p * diag(emis_p(:,val));
    val2(:, i) = max(val1(:, :, i));
end
% R to L
for i = nsize-1:-1:1
   val2_new = max(val1(:, :, i + 1), [], 2);
   val1(:, :, i) = val1(:, :, i) * diag(val2_new ./ val2(:, i));
    val2(:, i) = val2_new;
end
[~, V] = max(val2);
