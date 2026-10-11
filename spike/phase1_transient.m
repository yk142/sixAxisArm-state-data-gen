function T = phase1_transient()
%PHASE1_TRANSIENT 初期過渡の原因切り分け（issue #5）。6 軸、静止から重力補償＋手首ステップトルク。
%   変種: base / 摩擦なし / armature なし / 刻み 1/4 / 刻み 1/4 + armature なし
%   各変種の dq(t) を、対応する解析モデル（摩擦・armature の有無を合わせる）の ode45 参照と比べる。
c = sa_config();  p = c8_params(c.meshDir);  dd = fullfile(c.dataDir,'models');
q0 = [0.3 -0.8 0.9 0.2 0.5 -0.3]';
tg = c8RNEA(q0,zeros(6,1),zeros(6,1),p);  tau = tg + [0 0 0 0.5 1.0 2.0]';
tk = [1 2 5 10 25]*1e-3;
vars = {'base',   '',       struct(),                          1,   p; ...
        'noFric', '_nofr',  struct('friction',false,'suffix','_nofr'),  1, setfield(setfield(p,'frictionFc',0*p.frictionFc),'frictionBv',0*p.frictionBv); ...
        'noArm',  '_noarm', struct('armature',false,'suffix','_noarm'), 1, setfield(p,'armature',0*p.armature)};
rows = {};
for k = 1:size(vars,1)
    mdl = ['sa_plant6' vars{k,2}];
    if k > 1 && ~isfile(fullfile(dd,[mdl '.slx'])), phase0_build(6, dd, vars{k,3}); end
    rows = [rows; runVar(mdl, vars{k,1}, 1, q0, tau, vars{k,5}, tk, dd)]; %#ok<AGROW>
end
% 刻み 1/4（base と noArm）
rows = [rows; runVar('sa_plant6','base dt/4',4,q0,tau,p,tk,dd)];
rows = [rows; runVar('sa_plant6_noarm','noArm dt/4',4,q0,tau,vars{3,5},tk,dd)];
T = cell2table(rows,'VariableNames',{'variant','joint','err_1ms','err_2ms','err_5ms','err_10ms','err_25ms'});
disp(T);
end

function rows = runVar(mdl, label, div, q0, tau, pRef, tk, dd)
if ~bdIsLoaded(mdl), load_system(fullfile(dd,[mdl '.slx'])); end
fs0 = 8000;  old = get_param(mdl,'FixedStep');
set_param(mdl,'FixedStep',num2str(1/(fs0*div),'%.12g'));
restore = onCleanup(@() set_param(mdl,'FixedStep',old));
w = warning('off','all');  restoreW = onCleanup(@() warning(w));
t = (0:1/fs0:0.05)';
setSaInitialState(mdl, q0, zeros(6,1));
tt = reshape([t(1:end-1) t(2:end)-eps(t(2:end))]', [], 1);  uu = kron(repmat(tau',numel(t)-1,1), [1;1]);
assignin('base','SA_tau_in',[tt uu; t(end) tau']);
o = sim(mdl,'StopTime','0.05','LoadExternalInput','on','ExternalInput','SA_tau_in','SaveOutput','on','SaveFormat','Dataset');
ts = o.yout.getElement(2).Values.Time;  dq = o.yout.getElement(2).Values.Data;
f = @(~,x) [x(7:12); saFwdDyn(pRef,x(1:6),x(7:12),tau)];
[tr,xr] = ode45(f,[0 0.05],[q0;zeros(6,1)],odeset('RelTol',1e-10,'AbsTol',1e-12));
rows = cell(0,7);
for j = 4:6
    e = zeros(1,numel(tk));
    for i = 1:numel(tk)
        e(i) = interp1(ts,dq(:,j),tk(i)) - interp1(tr,xr(:,6+j),tk(i));
    end
    rows(end+1,:) = [{label, sprintf('J%d',j)}, num2cell(e)]; %#ok<AGROW>
end
end
