function r = saCoverage(ds, opts)
%SACOVERAGE 状態空間のカバレッジ指標（Statistics Toolbox を使わない自前実装）。
%   r = SACOVERAGE(ds, opts)   opts.nBins: 正規化 2*nAx 次元空間の 1 軸あたりの分割数（既定 [4 6 8]）
%                              opts.minCount: 「覆われた」とみなす train の最小点数（既定 [1 10]）
%   状態 x = [q/半可動幅, dq/qdMax]（q は可動範囲の中心からの正規化、[-1,1] に収まる）を格子に切り、
%   評価側（val / test / benchmark）の点のうち、同じ格子セルに train の点が minCount 以上あるものの割合を返す。
%   軸別（2 次元 q-dq）は 40 分割。
if nargin < 2, opts = struct(); end
if ~isfield(opts,'nBins'), opts.nBins = [4 6 8]; end
if ~isfield(opts,'minCount'), opts.minCount = [1 10]; end
nAx = ds.manifest.nAx;  sp = ds.manifest.params;
mid = (sp.qMax + sp.qMin)/2;  half = (sp.qMax - sp.qMin)/2;
sc = ds.scen;  splits = {sc.split};
nrm = @(s) [(s.q(1:end-1,:) - mid)./half, s.dq(1:end-1,:)./sp.qdMax];
X = struct();
for sn = {'train','val','test','benchmark'}
    parts = arrayfun(@(s) nrm(s), sc(strcmp(splits, sn{1})), 'UniformOutput', false);
    X.(sn{1}) = vertcat(parts{:});
end
r.n = structfun(@(a) size(a,1), X, 'UniformOutput', false);
% ---- 6 次元（2*nAx 次元）格子 ----
r.grid = struct('nBins',{},'minCount',{},'val',{},'test',{},'benchmark',{},'nTrainCells',{},'nCells',{});
for B = opts.nBins
    kt = cellIndex(X.train, B);
    [u, ~, ic] = unique(kt);  cnt = accumarray(ic, 1);
    for mc = opts.minCount
        okCells = u(cnt >= mc);
        g.nBins = B;  g.minCount = mc;
        for sn = {'val','test','benchmark'}
            g.(sn{1}) = mean(ismember(cellIndex(X.(sn{1}), B), okCells));
        end
        g.nTrainCells = numel(okCells);  g.nCells = B^(2*nAx);
        r.grid(end+1) = g; %#ok<AGROW>
    end
end
% ---- 軸別 q-dq 平面（40 分割）----
Bp = 40;  r.axisPlane = zeros(nAx, 3);  r.axisFill = zeros(nAx, 1);
for a = 1:nAx
    ktr = cellIndex(X.train(:,[a nAx+a]), Bp);
    [u, ~, ic] = unique(ktr);  cnt = accumarray(ic, 1);  okCells = u(cnt >= 1);
    for j = 1:3
        sn = {'val','test','benchmark'};  r.axisPlane(a,j) = mean(ismember(cellIndex(X.(sn{j})(:,[a nAx+a]), Bp), okCells));
    end
    r.axisFill(a) = numel(okCells)/Bp^2;
end
% ---- 速度の範囲: train の |dq|/qdMax の 99.9 パーセンタイル（sort で自前）と、それを超える評価側の点の割合 ----
as = sort(abs(X.train(:,nAx+1:end)), 1);
r.dqTrain999 = as(ceil(0.999*size(as,1)), :);  r.dqTrainMax = as(end, :);
for sn = {'val','test','benchmark'}
    r.dqOver.(sn{1}) = mean(abs(X.(sn{1})(:,nAx+1:end)) > r.dqTrain999, 1);
end
r.X = X;
end

function k = cellIndex(Z, B)
% [-1,1] を B 分割して格子セルの通し番号にする（範囲外は端のセルに入れる）
idx = min(max(floor((Z + 1)/2*B), 0), B-1);
d = size(Z, 2);
k = idx * (B.^(0:d-1))' + 1;
end
