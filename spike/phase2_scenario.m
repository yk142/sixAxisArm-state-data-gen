function S = phase2_scenario(nAx, T, seed)
%PHASE2_SCENARIO 取り決めの実験用シナリオ: 重力補償＋各軸 ±15% 定格の正弦和（5〜40 Hz）。8 kHz 格子の標本値。
c = sa_config();  p = c8_params(c.meshDir);
rs = RandStream('mt19937ar','Seed',seed);
lo = 0.5*p.qMin(1:nAx)';  hi = 0.5*p.qMax(1:nAx)';
S.q0 = lo + (hi-lo).*rand(rs, nAx, 1);
S.t = (0:1/c.fsSim:T)';
g = c8RNEA([S.q0; zeros(6-nAx,1)], zeros(6,1), zeros(6,1), p);  g = g(1:nAx);
tau = repmat(g', numel(S.t), 1);
for j = 1:nAx
    f = 5 + 35*rand(rs,1,3);  ph = 2*pi*rand(rs,1,3);
    tau(:,j) = tau(:,j) + 0.15*p.tauRated(j)*sum(sin(2*pi*S.t.*f + ph),2)/3;
end
S.tau = tau;  S.nAx = nAx;  S.fs = c.fsSim;
end
