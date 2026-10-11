function R = phase0_measure(nAx, qHold, outFile)
%PHASE0_MEASURE プラントの 8 kHz 実行時間・安定性・メモリを実測する。
%   R = PHASE0_MEASURE(nAx, qHold)   nAx=3|6。モデルは data/models/sa_plant<nAx>.slx。
%   試験: (a) RNEA 重力補償保持 2 s のドリフト (b) 無トルク自由落下 5 s (c) 重力補償+小励振 3 s の RNEA 整合
c = sa_config();  p = c8_params(c.meshDir);
mdl = sprintf('sa_plant%d', nAx);
if ~bdIsLoaded(mdl), load_system(fullfile(c.dataDir,'models',[mdl '.slx'])); end
fs = c.fsSim;  R = struct();
setPose = @(q0) arrayfun(@(i) set_param(sprintf('%s/joint%d',mdl,i), ...
    'PositionTargetSpecify','on','PositionTargetValue',sprintf('%.12g',q0(i)), ...
    'PositionTargetValueUnits','rad'), 1:nAx);
runit = @(T,U) sim(mdl,'StopTime',num2str(T),'LoadExternalInput','on', ...
    'ExternalInput','SA_in','SaveOutput','on','SaveFormat','Dataset');
qh = qHold(1:nAx)';

% (a) 保持
setPose(qh);  tg = c8RNEA([qh; zeros(6-nAx,1)], zeros(6,1), zeros(6,1), p);  tg = tg(1:nAx);
T = 2;  t = (0:1/fs:T)';  assignin('base','SA_in',[t repmat(tg',numel(t),1)]);
tic; o = runit(T); R.hold.wall = toc;
q = o.yout.getElement(1).Values.Data;  R.hold.driftDeg = rad2deg(max(abs(q - q(1,:)),[],1));
R.hold.finite = all(isfinite(q(:)));

% (b) 自由落下
setPose(qh);  T = 5;  t = (0:1/fs:T)';  assignin('base','SA_in',[t zeros(numel(t),nAx)]);
tic; o = runit(T); R.fall.wall = toc;
q = o.yout.getElement(1).Values.Data;  dq = o.yout.getElement(2).Values.Data;
R.fall.finite = all(isfinite(q(:))&isfinite(dq(:)));
R.fall.qMinDeg = rad2deg(min(q,[],1));  R.fall.qMaxDeg = rad2deg(max(q,[],1));
R.fall.qLimMinDeg = rad2deg(p.qMin(1:nAx));  R.fall.qLimMaxDeg = rad2deg(p.qMax(1:nAx));
R.fall.maxAbsDq = max(abs(dq),[],1);

% (c) 重力補償+各軸 +30 N*m ステップ 50 ms → 独立な前進動力学(RNEA+摩擦, ode45)との速度差
setPose(qh);  T = 0.05;  t = (0:1/fs:T)';  tauIn = tg + 30*ones(nAx,1);
assignin('base','SA_in',[t repmat(tauIn',numel(t),1)]);
tic; o = runit(T); R.exc.wall = toc;
dq = o.yout.getElement(2).Values.Data;  tt = o.yout.getElement(2).Values.Time;
[tr,xr] = ode45(@(~,x) fwdDyn(x,p,tauIn,nAx),[0 T],[qh;zeros(nAx,1)],odeset('RelTol',1e-9,'AbsTol',1e-11));
dqr = interp1(tr,xr(:,nAx+1:end),tt);
R.exc.dqSim_end = dq(end,:);  R.exc.dqRef_end = dqr(end,:);
R.exc.maxAbsDiff = max(abs(dq-dqr),[],1);  R.exc.finite = all(isfinite(dq(:)));  R.pose = rad2deg(qh');
if nargin>2, save(outFile,'R'); end
disp(R.hold); disp(R.fall); disp(R.exc);
end

function dx = fwdDyn(x,p,tau,n)
q = [x(1:n); zeros(6-n,1)];  dq = [x(n+1:2*n); zeros(6-n,1)];  z = zeros(6,1);
g = c8RNEA(q,z,z,p);  M = zeros(6);
for j=1:6, e=z; e(j)=1; M(:,j)=c8RNEA(q,z,e,p)-g; end
b = c8RNEA(q,dq,z,p)-g;  fr = p.frictionBv(:).*dq + p.frictionFc(:).*tanh(dq./p.frictionEps(:));
ddq = M(1:n,1:n) \ (tau - g(1:n) - b(1:n) - fr(1:n));
dx = [x(n+1:2*n); ddq];
end
