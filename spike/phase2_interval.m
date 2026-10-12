function R = phase2_interval(p, S, o)
%PHASE2_INTERVAL 1 kHz 遷移 x_k → x_{k+1}（8 ステップ）の予測誤差を、τ_k の定義別に比べる。
%   S: シナリオ（t, tau: 8 kHz 標本値）、o: プラントの格子出力（q, dq）。
%   定義: a 始点値 τ(t_k)  b 区間の台形平均（FOH 入力の平均）  c 左端 8 標本の単純平均  d 終点値 τ(t_{k+1})
%         e FOH 入力そのまま（解析モデルの積分。上限の目安）
n = S.nAx;  m = 8;  K = floor((numel(S.t)-1)/m);  fs = S.fs;
X = [o.q o.dq];  names = {'a start','b trapz mean','c left mean','d end','e FOH exact'};
err = zeros(K, n, numel(names));  d1 = zeros(K, n);
for k = 1:K
    i0 = (k-1)*m + 1;  i1 = i0 + m;
    seg = S.tau(i0:i1,:);
    tauSet = {seg(1,:), (sum(seg,1) - 0.5*(seg(1,:)+seg(end,:)))/m, mean(seg(1:m,:),1), seg(end,:)};
    x0 = X(i0,:)';  xt = X(i1,:)';
    for v = 1:4
        xp = rk4(p, x0, tauSet{v}', m, 1/fs, n);   err(k,:,v) = xp(n+1:end)' - xt(n+1:end)';
    end
    [~,xx] = ode45(@(s,y)[y(n+1:end); saFwdDyn(p,y(1:n),y(n+1:end),foh(seg,s*fs))], [0 m/fs], x0, odeset('RelTol',1e-10,'AbsTol',1e-12));
    err(k,:,5) = xx(end,n+1:end) - xt(n+1:end)';
    d1(k,:) = xt(n+1:end)' - x0(n+1:end)';
end
R.names = names;  R.rms = squeeze(sqrt(mean(err.^2,1)))';   % [5 x n]
R.dqStepRms = sqrt(mean(d1.^2,1));
end

function tau = foh(seg, u)   % u: 格子ステップ単位の位置 [0, m]
k = min(floor(u), size(seg,1)-2);  a = u - k;
tau = ((1-a)*seg(k+1,:) + a*seg(k+2,:))';
end

function x = rk4(p, x, tau, m, dt, n)
f = @(y) [y(n+1:end); saFwdDyn(p, y(1:n), y(n+1:end), tau)];
for i = 1:m
    k1 = f(x); k2 = f(x+0.5*dt*k1); k3 = f(x+0.5*dt*k2); k4 = f(x+dt*k3);
    x = x + dt/6*(k1+2*k2+2*k3+k4);
end
end
