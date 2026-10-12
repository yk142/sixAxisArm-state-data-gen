function T = saAnalyticResidual(ds, nPer, seed)
%SAANALYTICRESIDUAL 解析モデル（RNEA＋摩擦、saFwdDyn）との 1 ステップ整合をシナリオごとに集計する。
%   T = SAANALYTICRESIDUAL(ds, nPer, seed)  各シナリオから nPer 点（等間隔）を取り、加速度換算
%   a = Δdq/Δt と saFwdDyn の予測の差を返す。T は表（name, pattern, split, rmsRes[nAx], rmsAcc[nAx]）。
%   saFwdDyn は 1 回約 5〜10 ms なので全遷移ではなく抜き取り（実測は report に記載）。
if nargin < 2, nPer = 100; end
nAx = ds.manifest.nAx;  p = c8_params(sa_config().meshDir);  dt = 1/ds.manifest.fsData;
sc = ds.scen;  n = numel(sc);
rmsRes = zeros(n, nAx);  rmsAcc = zeros(n, nAx);
for k = 1:n
    s = sc(k);  K = s.nTransition;
    i = unique(round(linspace(1, K, min(nPer, K))));
    A = (s.dq(i+1,:) - s.dq(i,:))/dt;  P = zeros(numel(i), nAx);
    for j = 1:numel(i)
        P(j,:) = saFwdDyn(p, s.q(i(j),:)', s.dq(i(j),:)', s.tau(i(j),:)')';
    end
    rmsRes(k,:) = sqrt(mean((A - P).^2, 1));  rmsAcc(k,:) = sqrt(mean(A.^2, 1));
end
T = table({sc.name}', {sc.pattern}', {sc.split}', rmsRes, rmsAcc, 'VariableNames', {'name','pattern','split','rmsRes','rmsAcc'});
end
