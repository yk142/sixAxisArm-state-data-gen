% Phase 3-5 事前の実測: バリアを目標軸だけリミットの外側へ移して、定トルク往復・自由落下・パルスで接触を測る
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
clear functions;
nAx = 3;  sp = saParams(nAx);  tmp = tempname;  mkdir(tmp);
mE = buildSaExcitation(nAx, struct('modelName','sa_excite3_c35','slxPath',fullfile(tmp,'sa_excite3_c35.slx')));
p = c8_params(sa_config().meshDir);  z6 = zeros(6,1);
grav = @(q) subsref(c8RNEA([q(:); zeros(6-nAx,1)], z6, z6, p)', struct('type','()','subs',{{1:nAx}}));
dt = 1/8000;  tol = deg2rad(0.1);
R = struct('kind',{},'axis',{},'side',{},'amp',{},'tHit',{},'vHit',{},'pen',{},'fracContact',{},'fracDyn',{},'vRebound',{},'nHits',{},'maxdq',{});
q0base = [0.2 -0.5 0.8];
run1 = @(kind, a, side, amp, tau, q0, dq0, dur) runOne(mE, sp, kind, a, side, amp, tau, q0, dq0, dur, tol);
for a = 1:nAx
    for side = [-1 1]
        for amp = [0.3 0.6 1.0]
            dur = 3;  t = (0:dt:dur)';  q0 = q0base;
            g = grav(q0);  tau = repmat(g, numel(t), 1);
            tau(:,a) = g(a) + side*amp*sp.tauRated(a);
            R(end+1) = run1('drive', a, side, amp, tau, q0, zeros(1,nAx), dur); %#ok<SAGROW>
            fprintf('%s a=%d side=%+d amp=%.1f: tHit=%.2f vHit=%.2f pen=%.2f deg frac=%.2f dyn=%.3f hits=%d maxdq/qdMax=%.2f\n', 'drive', a, side, amp, R(end).tHit, R(end).vHit/sp.qdMax(a), R(end).pen, R(end).fracContact, R(end).fracDyn, R(end).nHits, R(end).maxdq/sp.qdMax(a));
        end
        % パルス: 重力＋±amp の矩形波（周期 1/f）
        for f = [1 2]
            amp = 0.6;  dur = 4;  t = (0:dt:dur)';  q0 = q0base;
            g = grav(q0);  tau = repmat(g, numel(t), 1);
            tau(:,a) = g(a) + side*amp*sp.tauRated(a)*sign(sin(2*pi*f*t + 0.01));
            R(end+1) = run1(sprintf('bounce%dHz',f), a, side, amp, tau, q0, zeros(1,nAx), dur); %#ok<SAGROW>
            fprintf('%s a=%d: tHit=%.2f vHit=%.2f pen=%.2f deg frac=%.2f dyn=%.3f hits=%d\n', R(end).kind, a, R(end).tHit, R(end).vHit/sp.qdMax(a), R(end).pen, R(end).fracContact, R(end).fracDyn, R(end).nHits);
        end
    end
end
% 自由落下（無トルク、全軸）: 姿勢をいくつか
for q0 = [0 -0.3 0.5; 0.8 -1.2 1.5; -0.8 -0.2 0.2; 0 -1.8 2.5; 0 1.0 -2.0]'
    dur = 3;  t = (0:dt:dur)';
    R(end+1) = run1('fall', 0, 0, 0, zeros(numel(t),nAx), q0', zeros(1,nAx), dur); %#ok<SAGROW>
    fprintf('fall q0=%s: tHit=%.2f vHit=%.2f pen=%.2f frac=%.2f dyn=%.3f\n', mat2str(q0',2), R(end).tHit, R(end).vHit, R(end).pen, R(end).fracContact, R(end).fracDyn);
end
save(fullfile(repo,'data','phase35_contact_spike.mat'),'R');
close_system(mE,0); rmdir(tmp,'s');

function r = runOne(mE, sp, kind, a, side, amp, tau, q0, dq0, dur, tol)
nAx = numel(q0);  t = (0:1/8000:dur)';
frac = 0.85*ones(1,nAx);  dfrac = 0.60*ones(1,nAx);
if a > 0, frac(a) = 1.10;  dfrac(a) = 0.80; else, frac(:) = 1.10;  dfrac(:) = 0.80; end
setSaBarrier(mE, sp, frac, dfrac);
o = simSa(mE, t, tau, q0', dq0');
% 目標軸（fall は全軸のうち最も接触した軸）
nearLo = o.q <= sp.qMin + tol;  nearHi = o.q >= sp.qMax - tol;  near = nearLo | nearHi;
if a == 0, [~, a] = max(sum(near,1)); end
qa = o.q(:,a);  dqa = o.dq(:,a);  n = near(:,a);
r.kind = kind;  r.axis = a;  r.side = side;  r.amp = amp;
i1 = find(n, 1);
if isempty(i1), r.tHit = NaN; r.vHit = NaN; else, r.tHit = t(i1); r.vHit = abs(dqa(i1)); end
r.pen = rad2deg(max([0; sp.qMin(a) - qa; qa - sp.qMax(a)]));
r.fracContact = mean(n);
dyn = n & (abs(dqa) > 0.02);  r.fracDyn = mean(dyn);
r.nHits = sum(diff([0; n]) == 1);
r.maxdq = max(abs(dqa));
r.vRebound = NaN;
end
