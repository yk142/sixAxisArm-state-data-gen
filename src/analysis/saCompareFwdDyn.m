function e = saCompareFwdDyn(mdl, p, q0, dq0, tau, T, fs, tWarm)
%SACOMPAREFWDDYN プラントと解析モデル(ode45)の同一入力での応答差。
%   e.dq, e.q: 軸ごとの最大絶対差（t=0 の初期状態から積分した参照との差）
%   e.dqW, e.qW: tWarm 以降のみ。参照の初期状態を t=tWarm のプラント状態に置き直して積分した差
%   （初期の過渡を除いた「短窓の一致」）。
if nargin < 8, tWarm = 0.01; end
n = numel(q0);  t = (0:1/fs:T)';
w = warning('off','all');  restore = onCleanup(@() warning(w));
o = simSa(mdl, t, tau(:)', q0, dq0);
f = @(~,x) [x(n+1:2*n); saFwdDyn(p,x(1:n),x(n+1:2*n),tau(:))];
op = odeset('RelTol',1e-10,'AbsTol',1e-12);
[tr,xr] = ode45(f, [0 T], [q0(:);dq0(:)], op);
xi = interp1(tr, xr, o.t);
e.dq = max(abs(o.dq - xi(:,n+1:end)),[],1);  e.q = max(abs(o.q - xi(:,1:n)),[],1);
k0 = find(o.t >= tWarm, 1);
[tr,xr] = ode45(f, o.t(k0:end)-o.t(k0), [o.q(k0,:) o.dq(k0,:)]', op);
e.dqW = max(abs(o.dq(k0:end,:) - xr(:,n+1:end)),[],1);  e.qW = max(abs(o.q(k0:end,:) - xr(:,1:n)),[],1);
e.scale = max(abs(xr(:,n+1:end)),[],1);  e.o = o;  e.xi = xi;
end
