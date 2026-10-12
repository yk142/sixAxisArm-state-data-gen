function S = phase2_scenario_1k(nAx, T, seed)
%PHASE2_SCENARIO_1K 1 kHz の制御器を想定: 各 1 ms の区間でトルクが一定（重力補償＋乱数 ±10% 定格）。
%   8 kHz 標本値は区間内で一定、境界の次の標本で新しい値（線形補間で 1 ステップかけて立ち上がる）。
c = sa_config();  p = c8_params(c.meshDir);
rs = RandStream('mt19937ar','Seed',seed);
lo = 0.5*p.qMin(1:nAx)';  hi = 0.5*p.qMax(1:nAx)';
S.q0 = lo + (hi-lo).*rand(rs, nAx, 1);
S.t = (0:1/c.fsSim:T)';
g = c8RNEA([S.q0; zeros(6-nAx,1)], zeros(6,1), zeros(6,1), p);  g = g(1:nAx);
K = ceil(T*1000);  u = g' + 0.10*p.tauRated(1:nAx).*(2*rand(rs,K,nAx)-1);
S.tau = u(min(floor((0:numel(S.t)-1)'/8)+1, K), :);
S.nAx = nAx;  S.fs = c.fsSim;
end
