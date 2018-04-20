% service latency of the network
% constraint of cloud offloading, near nodes offloading and local processing
% input: optimisation variable x at t, bt
% output:  constraint
function constraint = g(x, N, bt) %now just for M = 1
constraint = zeros(N,1);
for n=1:N
    zn = x(n, 1);
    if n==1
        ykn(1,1) = x(3*N,1);
        ykn(2,1) = x(N+3,1);
    elseif n==N
        ykn(1,1) = x(3*N-2,1);
        ykn(2,1) = x(N+1,1);
    else
        ykn(1,1) = x(N+2*n-2,1);
        ykn(2,1) = x(N+2*n+1,1);
    end
    ynk = x(N+2*n-1:N+2*n,1);
    ynn = x(3*N+n,1);
    constraint(n,1) = bt(n) + sum(ykn) - sum(ynk) - zn - ynn; %(2)
end
end