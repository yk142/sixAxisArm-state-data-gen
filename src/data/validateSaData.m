function r = validateSaData(ds, opts)
%VALIDATESADATA 組み立てたデータセットの検証（FR-5）。
%   r = VALIDATESADATA(ds, opts)
%     opts.nSample  解析モデルとの 1 ステップ整合に使う遷移数（既定 2000、シナリオ全体から等間隔に抽出）
%     opts.verbose  結果を表示（既定 true）
%   r.ok       すべての検査に合格したか、r.fail 不合格の検査（cell）
%   r.metrics  解析モデル（RNEA＋摩擦、saFwdDyn）との 1 ステップ整合の残差比（軸ごと。報告用で合否には使わない）
%   検査: 有限性、トルク ≤ tauPeak、角度が可動範囲 ± 1° 内、|dq| ≤ qdMax（接触シナリオは 1.3 倍まで）、配列サイズ、split のリーク
%   （名前・シードの重複なし、PTP は benchmark のみ、benchmark は PTP のみ）、初期状態の重複なし。
if nargin < 2, opts = struct(); end
if ~isfield(opts,'nSample'), opts.nSample = 2000; end
if ~isfield(opts,'verbose'), opts.verbose = true; end
sc = ds.scen;  nAx = ds.manifest.nAx;  sp = ds.manifest.params;
fail = {};
for k = 1:numel(sc)
    s = sc(k);  K = s.nTransition;  nm = s.name;
    fail = addIf(fail, isequal(size(s.q), [K+1 nAx]) && isequal(size(s.dq), [K+1 nAx]) && isequal(size(s.tau), [K nAx]) ...
        && isequal(size(s.tauInst), [K nAx]) && numel(s.atLimit) == K, sprintf('%s: 配列サイズ', nm));
    fail = addIf(fail, all(isfinite(s.q),'all') && all(isfinite(s.dq),'all') && all(isfinite(s.tau),'all'), sprintf('%s: 有限性', nm));
    fail = addIf(fail, all(abs(s.tau) <= sp.tauPeak*(1 + 1e-6), 'all'), sprintf('%s: トルクが tauPeak 超過', nm));
    fail = addIf(fail, all(s.q >= sp.qMin - deg2rad(1) & s.q <= sp.qMax + deg2rad(1), 'all'), sprintf('%s: 角度が可動範囲外', nm));
    dqTol = 1;  if strncmp(s.pattern, 'contact_', 8), dqTol = 1.3; end      % 接触シナリオは衝突の衝撃で他軸が一時的に qdMax を超えうる（docs/phase3_5_report.md）
    fail = addIf(fail, all(abs(s.dq) <= sp.qdMax*dqTol, 'all'), sprintf('%s: 角速度が qdMax 超過', nm));
end
% ---- split のリーク ----
fail = addIf(fail, numel(unique({sc.name})) == numel(sc), '名前の重複');
fail = addIf(fail, numel(unique([sc.seed])) == numel(sc), 'シードの重複');
isPtp = strcmp({sc.type},'ptp');  isBm = strcmp({sc.split},'benchmark');
fail = addIf(fail, all(isPtp == isBm), 'PTP と benchmark が一致しない');
fail = addIf(fail, all(ismember({sc.split}, {'train','val','test','benchmark'})), '未知の split');
ex = find(strcmp({sc.type},'excitation'));
if numel(ex) > 1
    q0 = cell2mat(arrayfun(@(k) [sc(k).q(1,:), sc(k).dq(1,:)], ex(:), 'UniformOutput', false));
    fail = addIf(fail, size(unique(round(q0, 9), 'rows'), 1) == numel(ex), '励振シナリオの初期状態が重複');
end
% ---- 解析モデルとの 1 ステップ整合（加速度換算）----
p = c8_params(sa_config().meshDir);  dt = 1/ds.manifest.fsData;
tot = sum([sc.nTransition]);  ia = round(linspace(1, tot, min(opts.nSample, tot)));
off = [0, cumsum([sc.nTransition])];
A = zeros(numel(ia), nAx);  P = A;
for j = 1:numel(ia)
    k = find(ia(j) > off, 1, 'last');  i = ia(j) - off(k);
    s = sc(k);
    A(j,:) = (s.dq(i+1,:) - s.dq(i,:)) / dt;
    P(j,:) = saFwdDyn(p, s.q(i,:)', s.dq(i,:)', s.tau(i,:)')';
end
r.metrics.accStd = std(A,0,1);
r.metrics.residRatio = sqrt(mean((A - P).^2, 1)) ./ r.metrics.accStd;      % 残差の RMS / 目標の標準偏差
r.metrics.nSample = numel(ia);
r.fail = fail;  r.ok = isempty(fail);
if opts.verbose
    fprintf('validateSaData: %d シナリオ、%d 遷移 → %s\n', numel(sc), tot, ternary(r.ok, '合格', sprintf('不合格 %d 件', numel(fail))));
    for i = 1:min(numel(fail), 20), fprintf('  NG: %s\n', fail{i}); end
    fprintf('  解析モデルとの 1 ステップ整合（加速度、%d 点）の残差比: %s\n', r.metrics.nSample, mat2str(r.metrics.residRatio, 3));
end
end

function fail = addIf(fail, ok, msg)
if ~ok, fail{end+1} = msg; end
end

function y = ternary(c, a, b)
if c, y = a; else, y = b; end
end
