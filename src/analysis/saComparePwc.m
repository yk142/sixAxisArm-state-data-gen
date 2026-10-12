function e = saComparePwc(mdl, p, q0, tauSeq, fs, holdSteps, onGrid)
if nargin < 7, onGrid = false; end
%SACOMPAREPWC 区間定数トルク列（holdSteps ステップごとに更新）でのプラントと解析モデルの差。
%   onGrid=true のとき、プラントへは格子点の標本値を与え、参照も格子間を線形補間(FOH)で積分する。
%   tauSeq [K x nAx]: 各区間のトルク。e.dq, e.q: 軸ごとの最大絶対差（初期状態から積分した参照との差）、
%   e.dqRms: dq 差の RMS、e.scale: 参照の dq の最大絶対値。
nAx = numel(q0);  K = size(tauSeq,1);  N = K*holdSteps;  t = (0:N)'/fs;
tauT = repelem(tauSeq, holdSteps, 1);  tauT = [tauT; tauT(end,:)];
w = warning('off','all');  restore = onCleanup(@() warning(w));
o = simSa(mdl, t, tauT, q0, zeros(nAx,1), onGrid);
if numel(o.t) ~= numel(t) || max(abs(o.t - t)) > 1e-12   % 可変刻みソルバ: 格子へ線形補間
    o.q = interp1(o.t, o.q, t);  o.dq = interp1(o.t, o.dq, t);
end
x = [q0(:); zeros(nAx,1)];  ref = zeros(N+1, 2*nAx);  ref(1,:) = x';
for k = 1:N
    tu = tauT(k,:)';  tn = tauT(k+1,:)';
    if onGrid, tf = @(s) tu + (tn - tu)*(s*fs); else, tf = @(s) tu; end   % onGrid: 格子間は線形補間(FOH)
    [~,xx] = ode45(@(s,y)[y(nAx+1:end); saFwdDyn(p,y(1:nAx),y(nAx+1:end),tf(s))], [0 1/fs], x, odeset('RelTol',1e-10,'AbsTol',1e-12));
    x = xx(end,:)';  ref(k+1,:) = x';
end
d = o.dq - ref(:,nAx+1:end);
e.dq = max(abs(d),[],1);  e.q = max(abs(o.q - ref(:,1:nAx)),[],1);
e.dqRms = sqrt(mean(d.^2,1));  e.scale = max(abs(ref(:,nAx+1:end)),[],1);
end
