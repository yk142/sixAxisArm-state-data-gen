function info = saExportFlat(ds, outFile, opts)
%SAEXPORTFLAT 組み立てたデータセットを、Python（h5py）から読める平坦な HDF5 に書き出す。
%   info = SAEXPORTFLAT(ds, outFile, opts)
%     ds       saAssembleDataset / genSaDataset の出力（data/sa_dataset_*.mat を load したものでも可）
%     outFile  出力先（.h5）
%     opts.dtype    'single'（既定）| 'double'。Δq・Δdq は倍精度で計算してから丸める
%     opts.deflate  gzip レベル 0〜9（既定 4）
%     opts.check    検証用に、各 split から等間隔に抜いた行を倍精度で <outFile>_check.mat（v7）にも保存（既定 true）
%     opts.nCheck   検証用の行数/split（既定 2000）
%   HDF5 の構成（h5py から見た形。MATLAB は列優先なので転置して書く。インデックスは 0 始まり）。d = nAx:
%     /<split>/X          (N, 3d)  [q(d) dq(d) tau(d)]（SI）     /<split>/Y   (N, 2d)  [Δq(d) Δdq(d)]（1 ステップ = 1 ms）
%     /<split>/X_spec     (N, 4d)  [sin q, cos q, dq/DQ_MAX, tau/TAU_MAX]  /<split>/Y_spec (N, 2d)  [Δq/D_Q_MAX, Δdq/D_DQ_MAX]
%     /<split>/scenario_id (N,) int32  /scenarios/* の添字      /<split>/step (N,) int32  シナリオ内の遷移番号
%     /<split>/at_limit   (N,) uint8   /<split>/qref (N, d)  遷移開始時刻の参照角（励振は NaN）
%     /<split>/payload (N,) 手先ペイロード [kg]（遷移ごと。シナリオの値）
%     /scenarios/{name,pattern,type,split}、/scenarios/payload（[kg]）、{seed, n_transition, start}  start = そのシナリオの先頭行（所属 split の配列内、0 始まり）
%        行は split 内でシナリオ順・step 順に並ぶので、シナリオ s は [start, start+n_transition) の連続区間
%     /stats/{x_mean,x_std,y_mean,y_std}（train の全遷移）、/scale/*、/physics/*（可動範囲・トルク・摩擦の係数）
%     ルート属性: manifest の主な項目（nAx, preset, scenario_hash, gen_clean, gen_repo, matlab など）と layout
if nargin < 3, opts = struct(); end
def = struct('dtype','single','deflate',4,'check',true,'nCheck',2000);
f = fieldnames(def);
for i = 1:numel(f)
    if ~isfield(opts,f{i}) || isempty(opts.(f{i})), opts.(f{i}) = def.(f{i}); end
end
[outDir,~,ext] = fileparts(outFile);
assert(strcmpi(ext,'.h5'), 'saExportFlat:ext', '.h5 のみ対応');
if ~isempty(outDir) && ~isfolder(outDir), mkdir(outDir); end
nAx = ds.manifest.nAx;  S = ds.scen;  sc = ds.scale;
cast = @(A) cast_(A, opts.dtype);
if isfile(outFile), delete(outFile); end
splits = {'train','val','test','benchmark'};
start = zeros(numel(S),1);  info = struct('file',outFile,'dtype',opts.dtype,'n',struct());
chk = struct();
for sp = splits
    sp = sp{1};  idx = find(strcmp({S.split}, sp));
    if isempty(idx), continue; end
    K = [S(idx).nTransition];  N = sum(K);  off = [0 cumsum(K)];  start(idx) = off(1:end-1);
    X = zeros(N,3*nAx);  Y = zeros(N,2*nAx);  Xs = zeros(N,4*nAx);  Ys = zeros(N,2*nAx);
    pld = zeros(N,1);  sid = zeros(N,1,'int32');  stp = zeros(N,1,'int32');  at = zeros(N,1,'uint8');  qr = nan(N,nAx);
    for j = 1:numel(idx)
        s = S(idx(j));  r = off(j)+1 : off(j+1);  k = s.nTransition;
        q = s.q(1:k,:);  dq = s.dq(1:k,:);
        X(r,:) = [q, dq, s.tau];
        Y(r,:) = [s.q(2:k+1,:) - q, s.dq(2:k+1,:) - dq];
        Xs(r,:) = [sin(q), cos(q), dq./sc.DQ_MAX, s.tau./sc.TAU_MAX];
        Ys(r,:) = [Y(r,1:nAx)./sc.D_Q_MAX, Y(r,nAx+1:end)./sc.D_DQ_MAX];
        if isfield(s,'payload'), pld(r) = s.payload; end   % 無ければ 0 kg
        sid(r) = idx(j)-1;  stp(r) = (0:k-1)';  at(r) = uint8(s.atLimit(:));
        if ~isempty(s.qref), qr(r,:) = s.qref(1:k,:); end
    end
    if ~isempty(S(idx(1)).qm)      % 関節弾性つき（issue #73）: モータ側の状態 [qm dqm] と次ステップとの差
        Xm = zeros(N,2*nAx);  Ym = zeros(N,2*nAx);
        for j = 1:numel(idx)
            s = S(idx(j));  r = off(j)+1 : off(j+1);  k = s.nTransition;
            Xm(r,:) = [s.qm(1:k,:), s.dqm(1:k,:)];  Ym(r,:) = [s.qm(2:k+1,:) - s.qm(1:k,:), s.dqm(2:k+1,:) - s.dqm(1:k,:)];
        end
        writeMat(outFile, ['/' sp '/Xm'], cast(Xm), opts, 'qm [rad] (link-side equivalent), dqm [rad/s]');
        writeMat(outFile, ['/' sp '/Ym'], cast(Ym), opts, 'dq_qm [rad], d_dqm [rad/s] per 1 ms step');
    end
    writeMat(outFile, ['/' sp '/X'],      cast(X),  opts, 'q [rad], dq [rad/s], tau [N*m]; each block has nAx columns');
    writeMat(outFile, ['/' sp '/Y'],      cast(Y),  opts, 'dq_q [rad] (q_next - q), d_dq [rad/s] (dq_next - dq), per 1 ms step');
    writeMat(outFile, ['/' sp '/X_spec'], cast(Xs), opts, 'sin q, cos q, dq/DQ_MAX, tau/TAU_MAX');
    writeMat(outFile, ['/' sp '/Y_spec'], cast(Ys), opts, 'd_q/D_Q_MAX, d_dq/D_DQ_MAX');
    writeMat(outFile, ['/' sp '/qref'],   cast(qr), opts, 'reference angle [rad] at the start of the step (hold/ptp scenarios), NaN otherwise');
    writeVec(outFile, ['/' sp '/payload'],     pld, opts, 'end-effector payload mass [kg] (scenario value)');
    writeVec(outFile, ['/' sp '/scenario_id'], sid, opts, '0-based index into /scenarios');
    writeVec(outFile, ['/' sp '/step'],        stp, opts, '0-based transition index within the scenario');
    writeVec(outFile, ['/' sp '/at_limit'],    at,  opts, '1 = limit-bound transition');
    info.n.(sp) = N;
    if opts.check
        ii = unique(round(linspace(1, N, min(opts.nCheck, N))));
        chk.([sp '_idx']) = ii(:) - 1;  chk.([sp '_X']) = X(ii,:);  chk.([sp '_Y']) = Y(ii,:);  chk.([sp '_scenario_id']) = double(sid(ii));
    end
end
writeStr(outFile, '/scenarios/name',    string({S.name}));
writeStr(outFile, '/scenarios/pattern', string({S.pattern}));
writeStr(outFile, '/scenarios/type',    string({S.type}));
writeStr(outFile, '/scenarios/split',   string({S.split}));
writeVec(outFile, '/scenarios/payload',      payloadOf(S), opts, 'payload mass [kg]');
writeVec(outFile, '/scenarios/seed',         int64([S.seed]'), opts, '');
writeVec(outFile, '/scenarios/n_transition', int32([S.nTransition]'), opts, '');
writeVec(outFile, '/scenarios/start',        int32(start), opts, 'first row within its split arrays (0-based)');
st = ds.stats;
for nm = {'xMean','x_mean';'xStd','x_std';'yMean','y_mean';'yStd','y_std'}'
    writeVec(outFile, ['/stats/' nm{2}], double(st.(nm{1})(:)), opts, '');
end
for nm = fieldnames(sc)', writeVec(outFile, ['/scale/' nm{1}], double(sc.(nm{1})(:)), opts, ''); end
sp_ = ds.manifest.params;
for nm = {'qMin','qMax','qdMax','tauRated','tauPeak','Mmean','Mmin','Mmax','armature','Fc','Bv','eps'}
    writeVec(outFile, ['/physics/' nm{1}], double(sp_.(nm{1})(:)), opts, '');
end
m = ds.manifest;
put = @(k, v) h5writeatt(outFile, '/', k, v);
put('nAx', double(nAx));  put('preset', char(m.preset));  put('created', char(m.created));
put('fs_data_hz', double(m.fsData));  put('fs_raw_hz', double(m.fsRaw));  put('scenario_hash', char(m.scenarioHash));
put('parent_commit', char(m.parentCommit));  put('repo_commit', char(m.repoCommit));  put('matlab', char(m.matlab));
put('gen_clean', double(m.genClean));  put('gen_repo', strjoin(string(m.genRepo), ';'));  put('dtype', opts.dtype);
put('layout', 'row = transition; datasets are (N, d) as seen from h5py; indices 0-based; rows are ordered by scenario then step within each split');
put('X_columns', 'q[nAx], dq[nAx], tau[nAx]');  put('Y_columns', 'dq_q[nAx], d_dq[nAx] (1 ms step)');
d = dir(outFile);  info.bytes = d.bytes;
if opts.check
    chk.scenario_name = {S.name}';  chk.nAx = nAx;
    save([outFile(1:end-3) '_check.mat'], '-struct', 'chk', '-v7');
end
end

function A = cast_(A, dtype)
if strcmp(dtype,'single'), A = single(A); else, A = double(A); end
end

function writeMat(file, path, A, opts, columns)
% A は (N, d)。MATLAB は列優先なので (d, N) に転置して書く → h5py からは (N, d) に見える
[N, d] = size(A);
if N == 0, error('saExportFlat:empty', '%s は空です', path); end
h5create(file, path, [d N], 'Datatype', class(A), 'ChunkSize', [d min(N, 65536)], 'Deflate', opts.deflate);
h5write(file, path, A');
if ~isempty(columns), h5writeatt(file, path, 'columns', columns); end
end

function writeVec(file, path, v, opts, note)
v = v(:);
if isempty(v), return; end
if numel(v) > 1 && opts.deflate > 0
    h5create(file, path, numel(v), 'Datatype', class(v), 'ChunkSize', min(numel(v), 65536), 'Deflate', opts.deflate);
else
    h5create(file, path, numel(v), 'Datatype', class(v));
end
h5write(file, path, v);
if ~isempty(note), h5writeatt(file, path, 'note', note); end
end

function writeStr(file, path, s)
h5create(file, path, numel(s), 'Datatype', 'string');
h5write(file, path, s(:));
end

function p = payloadOf(S)
p = zeros(numel(S),1);
if isfield(S,'payload'), p = double([S.payload]'); end
end
