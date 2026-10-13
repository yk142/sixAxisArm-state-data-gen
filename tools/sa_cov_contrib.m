function T = sa_cov_contrib(dsFile, outFile)
%SA_COV_CONTRIB 接触パターンごとの、PTP（benchmark）の 6 次元被覆への寄与（issue #29）。
%   接触を除いた train（3-3 と同じ集合）の被覆を基準に、各接触パターンを足したときの増分と、同じ点数のランダム抽出の増分を比べる。
%   sa_cov_contrib('data/sa_dataset_full3.mat', 'reports/issue-29/cov_contrib.mat')
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
ds = load(dsFile);  sp = ds.manifest.params;  sc = ds.scen;  nAx = ds.manifest.nAx;
mid = (sp.qMax + sp.qMin)/2;  half = (sp.qMax - sp.qMin)/2;
nrm = @(s) [(s.q(1:end-1,:) - mid)./half, s.dq(1:end-1,:)./sp.qdMax];
isC = strncmp({sc.pattern}, 'contact_', 8);  tr = strcmp({sc.split}, 'train');
get = @(m) catRows(arrayfun(@(s) nrm(s), sc(m), 'UniformOutput', false));
Xb = get(strcmp({sc.split}, 'benchmark'));  Xbase = get(tr & ~isC);
pats = unique({sc(isC & tr).pattern}, 'stable');  rs = RandStream('twister', 'Seed', 1);
rows = {};
for B = [4 6 8]
    cb = unique(cellIndex(Xbase, B));  base = mean(ismember(cellIndex(Xb, B), cb));
    rows(end+1,:) = {B, 'base（接触なし train）', size(Xbase,1), base, 0}; %#ok<AGROW>
    for p = pats
        Xp = get(tr & strcmp({sc.pattern}, p{1}));
        c = unique([cb; cellIndex(Xp, B)]);  rows(end+1,:) = {B, p{1}, size(Xp,1), mean(ismember(cellIndex(Xb,B), c)), 0}; %#ok<AGROW>
    end
    Xa = get(tr & isC);  c = unique([cb; cellIndex(Xa, B)]);
    rows(end+1,:) = {B, '接触 4 パターン全部', size(Xa,1), mean(ismember(cellIndex(Xb,B), c)), 0}; %#ok<AGROW>
end
T = cell2table(rows, 'VariableNames', {'nBins','set','nPoints','ptpCoverage','x'});  T.x = [];
disp(T);  if nargin > 1, save(outFile, 'T'); end
end
function k = cellIndex(Z, B)
idx = min(max(floor((Z + 1)/2*B), 0), B-1);
k = idx * (B.^(0:size(Z,2)-1))' + 1;
end
function X = catRows(c)
X = vertcat(c{:});
end
