% service latency of the network
% latency of cloud offloading, near nodes offloading and local processing
% input: optimisation variable x at t, pt
% output:  delay (loss)
function latency = f(x, N, pt) %now just for M = 1
lnk = 8/10;
lnn = 8/50;
latency = 0;
for n=1:N
    zn = x(n, 1);
    ynk = x(N+2*n-1:N+2*n,1);
    ynn = x(3*N+n,1);
    latency = latency + exp(pt(n)*zn) + lnk*sum(ynk) + lnn*(ynn.^2); %(34)
%    fx1=sum(exp(cloud(t).*z_action))+sum(L1*y1_action)+sum(L2*y2_action.^2); %cost function
end
end