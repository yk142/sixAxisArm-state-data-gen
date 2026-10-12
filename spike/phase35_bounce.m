% Phase 3-5 事前の実測 2: リミット直前から壁向きの初速で入る「衝突・反発」型（bounce）
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
clear functions;
nAx = 3;  sp = saParams(nAx);  tmp = tempname;  mkdir(tmp);
mE = buildSaExcitation(nAx, struct('modelName','sa_excite3_b35','slxPath',fullfile(tmp,'sa_excite3_b35.slx')));
p = c8_params(sa_config().meshDir);  z6 = zeros(6,1);  dt = 1/8000;  tol = deg2rad(0.1);
q0base = [0.2 -0.5 0.8];  R = [];
for a = 1:nAx
  for side = [-1 1]
    for vf = [0.3 0.6]
      for b = [0 0.15 -0.3]
        dur = 1.5;  t = (0:dt:dur)';  q0 = q0base;
        lim = ternary(side > 0, sp.qMax(a), sp.qMin(a));  q0(a) = lim - side*deg2rad(2);
        g = c8RNEA([q0(:); zeros(6-nAx,1)], z6, z6, p)';  g = g(1:nAx);
        tau = repmat(g, numel(t), 1);  tau(:,a) = g(a) + side*b*sp.tauRated(a);
        dq0 = zeros(1,nAx);  dq0(a) = side*vf*sp.qdMax(a);
        frac = 0.85*ones(1,nAx);  dfrac = 0.60*ones(1,nAx);  frac(a) = 1.10;  dfrac(a) = 0.80;
        setSaBarrier(mE, sp, frac, dfrac);
        o = simSa(mE, t, tau, q0', dq0');
        n = o.q(:,a) <= sp.qMin(a) + tol | o.q(:,a) >= sp.qMax(a) - tol;
        dyn = n & abs(o.dq(:,a)) > 0.02;
        r = [a side vf b mean(n) mean(dyn) sum(diff([0; n]) == 1) rad2deg(max([0; sp.qMin(a)-o.q(:,a); o.q(:,a)-sp.qMax(a)])) max(abs(o.dq(:,a)))/sp.qdMax(a)];
        R = [R; r]; %#ok<AGROW>
        fprintf('bounce a=%d side=%+d v=%.1f bias=%+.2f: frac=%.2f dyn=%.3f hits=%d pen=%.2f deg maxdq/qdMax=%.2f\n', r);
      end
    end
  end
end
save(fullfile(repo,'data','phase35_bounce_spike.mat'),'R');
close_system(mE,0); rmdir(tmp,'s');
disp('DONE');

function y = ternary(c,a,b)
if c, y = a; else, y = b; end
end
