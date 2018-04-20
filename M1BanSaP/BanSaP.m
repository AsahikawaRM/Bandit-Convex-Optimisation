% BanSaP method with one-point partial feedback
clear all
% =========== Initialize Parameters ========================
N = 5; % number of nodes
d = 1; % one dimensional case M=1

% computation limits
znmax = 100; 
ynkmax = 10;
ynnmax = 50;
MC = 20; % number of Monte Carlo realization
T=1:2000;

% ===========Iteration=============
for mc=1:MC
    latency=0;
    constraint=zeros(N,1);
    lambda = zeros(N,1); % first iterate lambda1 (Nx1 matrix)
    % define ut
    utmp=rand(4*N,1);
    ut=utmp/norm(utmp); % random vector ut drawn from the unit sphere (norm(ut)=1)
    % zero iteration
    zn = zeros(1,N);
    ynk = zeros(1,2*N);
    ynn = zeros(1,N);
    xhat = [zn, ynk, ynn]';
    
    % q(n)
    for n = 1:N
        if n==1 || n==2 || n==3
            q(n)=unifrnd(32,40);
        elseif n==4 || n==5 
            q(n)=unifrnd(20,25);
        else
            q(n)=unifrnd(40,50);
        end
    end
    
    for t = 1:length(T) % 7.5 min a slot in 24h (192 times a day)
        disp(['Iteration'  num2str(t) '/' num2str(mc)]);
        
        % Step 0: Set parameters
        % Set stepsize
        alpha = 0.012*t.^(-0.75); % stepsize of xhat (alpha = O(T.^(-3/4)))
        mu = 0.54*t.^(-0.75); % stepsize of update of lambda (mu = O(T.^(-3/4)))
        delta = 20*t.^(-0.25); % when M = 1 (delta = O(T.^(-1/4)))
        gamma = 0.05*t.^(-0.25); % when M = 1 (gamma = delta / r)
        for n = 1:N
            % Set vt(n)
            if n==1 || n==2 || n==3
                vt(n)=unifrnd(36,44);
            elseif n==4 || n==5 
                vt(n)=unifrnd(22.5,27.5);
            else
                vt(n)=unifrnd(45,55);
            end
            
            % Set bt(n) (arrival rate at each t)
            bt(n)=q(n)*sin(pi*t/96)+vt(n);  % bt(n) is arrival rate
            
            % Set pt(n) (one node feedback case)
            if n==4 || n==5
                pt(n) = 0.045*sin(pi*t/96)+0.15;
            else
                pt(n) = 0.015*sin(pi*t/96)+0.05;
            end
        end
%-------------------------------------------------------------------------%
        % Step 1: The learner plays the perturbed actions
        x1 = xhat+delta*ut; % learner's action in bandit case (m from 1 to M, M=1)
        
        % Set znhat, ynkhat and ynnhat
        znhat = xhat(1:N, 1); % N elements
        ynkhat = xhat(N+1:3*N,1); % 2N elements
        ynnhat = xhat(3*N+1:4*N,1); % N elements
%-------------------------------------------------------------------------%    
        % Step 2: The nature reveals the losses and the constraint
        % calculate latency
        latency1 = f(x1, N, pt); % M=1, so only one loss(ft(x1,t))
        latency0 = f(xhat, N, pt);
        % calculate average loss
        latency = latency +latency0;
        Loss(t,mc) = latency/t;  % average latency
        
        % calculate constraint and dynamic fit
        constraint0=g(x1, N, bt);
        constraint=constraint+constraint0; % <=0, from 0 to T
        Fit(t,mc)=norm(max(constraint,0));
%-------------------------------------------------------------------------%
        %Step 3: Update the primal variable xhat(t+1)
        for n = 1:N
            % initialization
            uznhat = ut(n,1);
            uynkhat = ut(N+2*n-1:N+2*n,1);
            uynnhat = ut(3*N+n,1);
        
            % define k (two output nodes)
            if n==1
                k1 = N;
                k2 = 2;
            elseif n==N
                k1 = N-1;
                k2 = 1;
            else
                k1 = n-1;
                k2 = n+1;
            end
            grad_znhat = d/delta*latency1*uznhat; % latency0 = f(xhat+delta*ut)
            grad_ynkhat1 = d/delta*latency1*uynkhat(1,1);
            grad_ynkhat2 = d/delta*latency1*uynkhat(2,1);
            grad_ynnhat = d/delta*latency1*uynnhat;
            znhat(n,1) = min(100*(1-gamma) , max(znhat(n,1) - alpha*(grad_znhat-lambda(n)),0)); % (31a) (f(x1) is loss) function projection into [0, znmax]
            ynkhat(2*n-1,1) = min(20*(1-gamma) , max(ynkhat(2*n-1,1) - alpha*(grad_ynkhat1-lambda(n)+lambda(k1)),0)); % (31b) projection into [0, ynkmax]
            ynkhat(2*n,1) = min(20*(1-gamma) , max(ynkhat(2*n,1)-alpha*(grad_ynkhat2-lambda(n)+lambda(k2)),0)); % (31b) projection into [0, ynkmax]
            ynnhat(n,1) = min(50*(1-gamma) , max(ynnhat(n,1)-alpha*(grad_ynnhat-lambda(n)),0)); % (31c) projection into [0, ynnmax]
        end
    
        % update xhat
        xhat = [znhat;ynkhat;ynnhat];
%-------------------------------------------------------------------------%
        %Step 4: Update the dual variable lambda(t+1)
        % update lambda
        for n=1:N
            if n==1
                lambda(n) = max(lambda(n) + mu*(bt(n)+(ynkhat(2*N,1)+ynkhat(3,1))-sum(ynkhat(2*n-1:2*n,1))-znhat(n,1)-ynnhat(n,1)),0); % (33)
            elseif n ==N
                lambda(n) = max(lambda(n) + mu*(bt(n)+(ynkhat(2*N-1,1)+ynkhat(1,1))-sum(ynkhat(2*n-1:2*n,1))-znhat(n,1)-ynnhat(n,1)),0); % (33)
            else
                lambda(n) = max(lambda(n) + mu*(bt(n)+(ynkhat(2*n-2,1)+ynkhat(2*n+1,1))-sum(ynkhat(2*n-1:2*n,1))-znhat(n,1)-ynnhat(n,1)),0); % (33)
            end
        end
    end
end

Fitavg=mean(Fit,2);
Lossavg=mean(Loss,2);

a = [1:length(T)];
figure(1);
plot(a, Fitavg,'--b');
axis([0,length(T),0,10000]);
title("Dynamic fit");
figure(2);
plot(a, Lossavg,'--b');
axis([0,length(T),0,1000]);
title("Time-average cost");
