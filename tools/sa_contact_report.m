function out = sa_contact_report(dsFile, outDir)
%SA_CONTACT_REPORT 接触シナリオの評価（件数の内訳・衝突速度・めり込み・解析モデルとの整合）と図。
%   sa_contact_report('data/sa_dataset_full3.mat', 'reports/issue-27')
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
if ~isfolder(outDir), mkdir(outDir); end
ds = load(dsFile);  sp = ds.manifest.params;  sc = ds.scen;  nAx = ds.manifest.nAx;
r = saContactStats(ds);
% ---- 解析モデルとの 1 ステップ整合: 接触遷移 vs 非接触遷移（接触シナリオ内と、接触なしの 158 本）----
p = c8_params(sa_config().meshDir);  dt = 1/ds.manifest.fsData;  tol = deg2rad(0.1);  rs = RandStream('twister','Seed',1);
resid = struct();
cats = {'contact_in_contact_scen','free_in_contact_scen','free_other_scen'};
smp = struct(cats{1},[],cats{2},[],cats{3},[]);
for k = 1:numel(sc)
    s = sc(k);  K = s.nTransition;  near = s.q <= sp.qMin + tol | s.q >= sp.qMax - tol;  nearT = any(near(1:K,:) | near(2:K+1,:), 2);
    isC = strncmp(s.pattern, 'contact_', 8);
    for c = 1:3
        if c == 1, idx = find(nearT); ok = isC; elseif c == 2, idx = find(~nearT); ok = isC; else, idx = find(~nearT); ok = ~isC; end
        if ~ok || isempty(idx), continue; end
        n = min(numel(idx), ternary(c == 3, 12, 40));  idx = idx(randperm(rs, numel(idx), n));
        for i = idx(:)'
            a = (s.dq(i+1,:) - s.dq(i,:))/dt;  pr = saFwdDyn(p, s.q(i,:)', s.dq(i,:)', s.tau(i,:)')';
            smp.(cats{c}) = [smp.(cats{c}); a pr];
        end
    end
end
for c = 1:3
    A = smp.(cats{c})(:,1:nAx);  P = smp.(cats{c})(:,nAx+1:end);
    resid.(cats{c}) = struct('n', size(A,1), 'ratio', sqrt(mean((A-P).^2,1))./sqrt(mean(A.^2,1)));
end
% ---- 図 1: パターン別の接触遷移（静止と動的）----
isCp = strncmp(r.patterns, 'contact_', 8);  pn = r.patterns(isCp);
tot = sum(r.axisCount(isCp,:), 2);  dyn = sum(r.axisDyn(isCp,:), 2);
f = figure('Visible','off','Position',[0 0 1000 420]);
subplot(1,2,1); bar(categorical(pn, pn), [dyn, tot - dyn], 'stacked'); set(gca,'TickLabelInterpreter','none'); grid on;
legend({'動的（衝突・反発・離脱）','静止して押し付け'}); ylabel('接触遷移数（軸別の合計）'); title('パターン別の接触遷移');
subplot(1,2,2); bar(categorical(arrayfun(@(a) sprintf('J%d',a), 1:nAx, 'UniformOutput', false)), [sum(r.axisDyn(isCp,:),1)', sum(r.axisCount(isCp,:),1)' - sum(r.axisDyn(isCp,:),1)'], 'stacked');
grid on; ylabel('接触遷移数'); title('軸別（接触シナリオ）'); legend({'動的','静止'});
exportgraphics(f, fullfile(outDir,'contact_counts.png'), 'Resolution', 110); close(f);
% ---- 図 2: 衝突速度の分布 ----
f = figure('Visible','off','Position',[0 0 900 380]);
for a = 1:nAx
    subplot(1,nAx,a);  h = r.hits([r.hits.axis] == a);
    if ~isempty(h), histogram([h.speed], 0:0.5:max(5, ceil(max([h.speed])))); end
    grid on; xlabel('衝突速度 |dq| [rad/s]'); ylabel('衝突回数'); title(sprintf('J%d（qdMax %.2f）', a, sp.qdMax(a)));
end
exportgraphics(f, fullfile(outDir,'impact_speed.png'), 'Resolution', 110); close(f);
% ---- 図 3: 代表シナリオの時系列とリミット近傍の位相平面 ----
names = {'contact_fall_001','contact_drive_002','contact_bln_003','contact_bounce_002'};
f = figure('Visible','off','Position',[0 0 1400 600]);
for j = 1:numel(names)
    s = sc(strcmp({sc.name}, names{j}));  t = (0:s.nTransition)'/ds.manifest.fsData;
    [~, a] = max(sum(s.q <= sp.qMin + tol | s.q >= sp.qMax - tol, 1));
    subplot(2,4,j);  plot(t, rad2deg(s.q(:,a)), 'b'); hold on; yline(rad2deg(sp.qMax(a)),'r--'); yline(rad2deg(sp.qMin(a)),'r--'); grid on;
    xlabel('t [s]'); ylabel(sprintf('q_%d [deg]', a)); title(names{j}, 'Interpreter','none');
    subplot(2,4,4+j); plot(t, s.dq(:,a), 'b'); grid on; xlabel('t [s]'); ylabel(sprintf('dq_%d [rad/s]', a));
end
exportgraphics(f, fullfile(outDir,'contact_timeseries.png'), 'Resolution', 100); close(f);
out = struct('stats', rmfield(r,'hits'), 'nHits', numel(r.hits), 'hits', r.hits, 'resid', resid);
save(fullfile(outDir,'contact_metrics.mat'), '-struct', 'out');
fprintf('パターン別 接触遷移（軸別）と動的:\n'); for i = 1:numel(r.patterns), if isCp(i), fprintf('  %-16s 遷移 %6d 接触 %s 動的 %s\n', r.patterns{i}, r.nTrans(i), mat2str(r.axisCount(i,:)), mat2str(r.axisDyn(i,:))); end, end
fprintf('合計: 接触 %d、動的 %d（%.1f%%）。衝突イベント %d 回。めり込み最大 下限 %s 上限 %s deg。|dq|>qdMax の遷移 %s（%d シナリオ）\n', ...
    sum(tot), sum(dyn), 100*sum(dyn)/sum(tot), numel(r.hits), mat2str(r.pen(1,:),2), mat2str(r.pen(2,:),2), mat2str(r.dqOver), r.dqOverScen);
for c = 1:3, fprintf('解析整合 %s: n=%d 残差比 %s\n', cats{c}, resid.(cats{c}).n, mat2str(resid.(cats{c}).ratio,3)); end
end

function y = ternary(c, a, b)
if c, y = a; else, y = b; end
end
