function r = phase1_kick(mdl, p, q0, tau, T)
%PHASE1_KICK t=T での dq(J4..J6) の、解析モデル(ode45)に対する差（初期の速度ジャンプの大きさ）。
%   p: 参照に使う c8_params（摩擦・armature の有無をモデルに合わせる）
if nargin < 5, T = 0.002; end
fs = 8000;  t = (0:1/fs:T)';
w = warning('off','all');  restore = onCleanup(@() warning(w));
o = simSa(mdl, t, tau(:)', q0, zeros(6,1));
f = @(~,x) [x(7:12); saFwdDyn(p,x(1:6),x(7:12),tau(:))];
[~,xr] = ode45(f, o.t, [q0(:);zeros(6,1)], odeset('RelTol',1e-10,'AbsTol',1e-12));
r = o.dq(end,4:6) - xr(end,10:12);
end
