function ds = saAssembleDataset(S, loadFn, nAx, opts)
%SAASSEMBLEDATASET シナリオ表とキャッシュ（8 kHz 生データ）から 1 kHz のデータセットを組み立てる。
%   ds = SAASSEMBLEDATASET(S, loadFn, nAx, opts)
%     S       saScenarioSet の要素（選んだ部分集合でもよい）
%     loadFn  @(s) キャッシュ構造体 e（q dq tau qref dqref q0 dq0 contact8k fs wall seed）を返す関数
%     opts    .preset（記録用）
%   ds.scen（シナリオごと）、ds.stats（train のみの正規化統計）、ds.scale、ds.manifest。
%   遷移 k: 入力 x_k = (q(k), dq(k), tau(k)) → 次状態 (q(k+1), dq(k+1))、Δt = 1 ms。
if nargin < 4, opts = struct(); end
if ~isfield(opts,'preset'), opts.preset = ''; end
c = sa_config();  sp = saParams(nAx);  tol = deg2rad(0.1);
scen = repmat(struct('name','','pattern','','type','','phase',[],'split','','seed',0,'q',[],'dq',[], ...
    'tau',[],'tauInst',[],'qref',[],'dqref',[],'atLimit',[],'contact8k',0,'nTransition',0), numel(S), 1);
for k = 1:numel(S)
    s = S(k);  e = loadFn(s);
    assert(e.seed == s.seed && strcmp(e.name, s.name), 'saAssembleDataset:cache', ...
        'キャッシュがシナリオ表と一致しません: %s', s.name);
    D = saDownsample(e, c.fsSim, c.fsData);
    lim = any(D.q <= sp.qMin + tol | D.q >= sp.qMax - tol, 2);       % リミット拘束（0.1° 以内または外側）
    atLimit = lim(1:end-1) | lim(2:end);
    scen(k) = struct('name',s.name,'pattern',s.pattern,'type',s.type,'phase',s.phase,'split',s.split,'seed',s.seed, ...
        'q',D.q,'dq',D.dq,'tau',D.tau,'tauInst',D.tauInst,'qref',D.qref,'dqref',D.dqref, ...
        'atLimit',atLimit,'contact8k',e.contact8k,'nTransition',size(D.tau,1));
end
% ---- 正規化統計: train の全遷移のみ（val / test / benchmark のリーク防止）----
isTrain = strcmp({scen.split},'train');
n = 0;  sx = zeros(1,3*nAx);  sxx = sx;  sy = zeros(1,2*nAx);  syy = sy;       % 逐次の和（連結せずに平均・標準偏差）
for k = find(isTrain)
    sc = scen(k);
    X = [sc.q(1:end-1,:), sc.dq(1:end-1,:), sc.tau];  Y = [diff(sc.q,1,1), diff(sc.dq,1,1)];
    n = n + size(X,1);  sx = sx + sum(X,1);  sxx = sxx + sum(X.^2,1);  sy = sy + sum(Y,1);  syy = syy + sum(Y.^2,1);
end
stats = struct('xMean',[],'xStd',[],'yMean',[],'yStd',[]);
if n > 1
    stats.xMean = sx/n;  stats.xStd = sqrt(max(sxx/n - stats.xMean.^2, 0)*n/(n-1));
    stats.yMean = sy/n;  stats.yStd = sqrt(max(syy/n - stats.yMean.^2, 0)*n/(n-1));
    stats.xStd(stats.xStd < eps) = 1;  stats.yStd(stats.yStd < eps) = 1;
end
stats.nTrain = n;
stats.note = '正規化統計は train の全遷移から算出。X=[q dq tau]（各 nAx 列）、Y=[Δq Δdq]';
dt = 1/c.fsData;
scale.DQ_MAX = sp.qdMax;  scale.TAU_MAX = sp.tauPeak;  scale.D_Q_MAX = sp.qdMax*dt;
scale.D_DQ_MAX = (sp.tauPeak ./ sp.Mmin)*dt;

m.created = char(datetime('now'));  m.preset = opts.preset;  m.nAx = nAx;
m.fsRaw = c.fsSim;  m.fsData = c.fsData;  m.nScenario = numel(scen);
m.nTrain = nnz(isTrain);  m.nVal = nnz(strcmp({scen.split},'val'));
m.nTest = nnz(strcmp({scen.split},'test'));  m.nBenchmark = nnz(strcmp({scen.split},'benchmark'));
m.nTransition = sum([scen.nTransition]);
m.names = {S.name};  m.seeds = [S.seed];  m.durations = [S.duration];
m.scenarioHash = hashOf([m.seeds, m.durations]);
m.downsample = '状態は瞬時値の間引き、トルクは区間の台形平均（saDownsample）';
m.params = sp;
m.parentCommit = gitHead(c.parentDir);  m.repoCommit = gitHead(c.rootDir);
m.matlab = version;
ds.scen = scen;  ds.stats = stats;  ds.scale = scale;  ds.manifest = m;
end

function h = hashOf(x)
h = '';
try
    md = java.security.MessageDigest.getInstance('SHA-256');
    md.update(typecast(double(x(:)'), 'uint8'));
    h = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'))', 1, []));
catch
end
end

function h = gitHead(d)
h = '';
[st, out] = system(sprintf('git -C "%s" rev-parse HEAD', d));
if st == 0, h = strtrim(out); end
end
