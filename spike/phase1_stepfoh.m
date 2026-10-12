function r = phase1_stepfoh(mdl, p, q0, tg, T)
%PHASE1_STEPFOH phase1_stepcheck と同じ試験（t=10 ms に J6 +2 N*m）を、入力を 8 kHz 格子の線形補間（1 ステップで立ち上がる）で与える。
%   eps 幅のステップ（simSa の厳密ゼロ次保持）は微分が非有限になるため、その切り分け用。
if nargin < 5, T = 0.02; end
fs = 8000;  t = (0:1/fs:T)';
tauT = repmat(tg(:)', numel(t), 1);  tauT(t >= 0.01, 6) = tauT(t >= 0.01, 6) + 2;
w = warning('off','all');  restore = onCleanup(@() warning(w));
setSaInitialState(mdl, q0, zeros(6,1));
assignin('base','SA_tau_in',[t tauT]);
o = sim(mdl,'StopTime',num2str(T),'LoadExternalInput','on','ExternalInput','SA_tau_in','SaveOutput','on','SaveFormat','Dataset');
dq = o.yout.getElement(2).Values.Data;  ts = o.yout.getElement(2).Values.Time;
x = [q0(:); zeros(6,1)];  ref = zeros(numel(t),12);  ref(1,:) = x';
for k = 1:numel(t)-1
    tu = tauT(k,:)';
    [~,xx] = ode45(@(~,y)[y(7:12); saFwdDyn(p,y(1:6),y(7:12),tu)], [0 1/fs], x, odeset('RelTol',1e-10,'AbsTol',1e-12));
    x = xx(end,:)';  ref(k+1,:) = x';
end
d = dq(:,6) - ref(:,12);
r.diff_end = d(end);  r.maxAbs = max(abs(d));  r.n = numel(ts);
end
