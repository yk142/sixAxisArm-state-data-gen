function r = phase1_stepcheck(mdl, p, q0, tg, T)
%PHASE1_STEPCHECK 重力補償保持の途中(t=10 ms)で J6 に +2 N*m のトルクステップを入れ、解析モデルとの dq6 差を返す。
%   r.diff_t0 : 先頭の過渡（t=0 のステップ）後 t=T までの dq6 差の最大、r.diff_step: 10 ms ステップ後の dq6 差の最終値
if nargin < 5, T = 0.02; end
fs = 8000;  t = (0:1/fs:T)';
tauT = repmat(tg(:)', numel(t), 1);  tauT(t >= 0.01, 6) = tauT(t >= 0.01, 6) + 2;
w = warning('off','all');  restore = onCleanup(@() warning(w));
o = simSa(mdl, t, tauT, q0, zeros(6,1));
x = [q0(:); zeros(6,1)];  ref = zeros(numel(t),12);  ref(1,:) = x';
for k = 1:numel(t)-1
    tu = tauT(k,:)';
    [~,xx] = ode45(@(~,y)[y(7:12); saFwdDyn(p,y(1:6),y(7:12),tu)], [0 1/fs], x, odeset('RelTol',1e-10,'AbsTol',1e-12));
    x = xx(end,:)';  ref(k+1,:) = x';
end
d = o.dq(:,6) - ref(:,12);
r.diff_before = d(find(t < 0.01, 1, 'last'));   % ステップ直前
r.diff_end = d(end);                            % T 時点
r.maxAbs = max(abs(d));
end
