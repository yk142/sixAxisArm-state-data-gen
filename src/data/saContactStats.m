function r = saContactStats(ds)
%SACONTACTSTATS リミット接触の統計（軸別・パターン別）。
%   r = SACONTACTSTATS(ds)
%   軸 a の「接触遷移」: 遷移の開始または終了時に q_a がリミットから 0.1° 以内か外側。
%   「動的な接触」: 接触遷移のうち |dq_a| > 0.02 rad/s（開始時）または |Δdq_a| > 0.005 rad/s（J2 版 docs/limit_contact_report.md と同じ定義）。
%   r.tab: パターン別（遷移数、軸ごとの接触遷移数・動的接触数）、r.hits: 衝突イベント（非接触 → 接触）ごとの軸・衝突速度・パターン、
%   r.pen: 軸ごとのめり込みの最大 [deg]、r.dqOver: |dq| > qdMax の遷移数（軸別、接触シナリオのみ）。
nAx = ds.manifest.nAx;  sp = ds.manifest.params;  sc = ds.scen;  tol = deg2rad(0.1);
pats = unique({sc.pattern}, 'stable');
r.axisCount = zeros(numel(pats), nAx);  r.axisDyn = zeros(numel(pats), nAx);  r.nTrans = zeros(numel(pats), 1);
r.hits = struct('pattern',{},'name',{},'axis',{},'speed',{},'side',{});
r.pen = zeros(2, nAx);  r.dqOver = zeros(1, nAx);  r.dqOverScen = 0;
for k = 1:numel(sc)
    s = sc(k);  K = s.nTransition;  ip = find(strcmp(pats, s.pattern));
    r.nTrans(ip) = r.nTrans(ip) + K;
    q = s.q;  dq = s.dq;
    near = q <= sp.qMin + tol | q >= sp.qMax - tol;               % [K+1 x nAx] 各時刻
    nearT = near(1:K,:) | near(2:K+1,:);
    dyn = nearT & (abs(dq(1:K,:)) > 0.02 | abs(diff(dq,1,1)) > 0.005);
    r.axisCount(ip,:) = r.axisCount(ip,:) + sum(nearT, 1);
    r.axisDyn(ip,:) = r.axisDyn(ip,:) + sum(dyn, 1);
    r.pen(1,:) = max(r.pen(1,:), rad2deg(max(sp.qMin - q, [], 1)));       % 下限側のめり込み
    r.pen(2,:) = max(r.pen(2,:), rad2deg(max(q - sp.qMax, [], 1)));       % 上限側
    for a = 1:nAx
        e = find(near(2:end,a) & ~near(1:end-1,a)) + 1;                  % 非接触 → 接触になった時刻
        for i = e(:)'
            r.hits(end+1) = struct('pattern',s.pattern,'name',s.name,'axis',a,'speed',abs(dq(i,a)),'side',sign(q(i,a) - (sp.qMax(a)+sp.qMin(a))/2)); %#ok<AGROW>
        end
    end
    if strncmp(s.pattern, 'contact_', 8)
        over = abs(dq) > sp.qdMax;  r.dqOver = r.dqOver + sum(over, 1);  r.dqOverScen = r.dqOverScen + any(over(:));
    end
end
r.patterns = pats;
end
