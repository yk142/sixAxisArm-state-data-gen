function ref = phase2_ref(p, S)
%PHASE2_REF 8 kHz 格子の標本値トルクを格子間で線形補間(FOH)して解析モデルを ode45 で積分（格子上の x を返す）。
n = S.nAx;  N = numel(S.t)-1;  fs = S.fs;
x = [S.q0; zeros(n,1)];  ref = zeros(N+1, 2*n);  ref(1,:) = x';
op = odeset('RelTol',1e-10,'AbsTol',1e-12);
for k = 1:N
    tu = S.tau(k,:)';  tn = S.tau(k+1,:)';
    [~,xx] = ode45(@(s,y)[y(n+1:end); saFwdDyn(p,y(1:n),y(n+1:end),tu+(tn-tu)*(s*fs))], [0 1/fs], x, op);
    x = xx(end,:)';  ref(k+1,:) = x';
end
end
