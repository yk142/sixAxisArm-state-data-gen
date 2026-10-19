function ds = genSaDataset(opts)
%GENSADATASET シナリオを実行して 6 軸アームの学習データセット（3 軸または 6 軸）を生成する。
%   ds = GENSADATASET(opts)
%     opts.nAx       3（既定）| 6
%     opts.preset    'full'（既定）| 'small'
%     opts.names     シナリオ名の cell（指定すると、その名前のシナリオだけを対象にする。動作確認・テスト用）
%     opts.outDir    保存先（既定 <repo>/data）。cache/ とデータセット .mat を置く
%     opts.shard     [i n]: n 分割したうちの i 番目だけ新規生成する（複数の MATLAB プロセスで分担）
%     opts.maxScen   1 回の呼び出しで新規生成するシナリオ数の上限（既定 Inf）
%     opts.cacheOnly true なら生成だけ行い、組み立てはしない（分担実行用）
%     opts.save      データセットを .mat に保存するか（既定 true）
%     opts.replicas  ペイロードを変えた複製の組数（既定 0。issue #65。saScenarioSetPayload。保存名は sa_dataset_<preset>pl<nAx>.mat）
%     opts.payloadGrid  複製に割り当てるペイロード [kg]（既定 [1 2 4 6 8]）
%   戻り値 ds: 全シナリオのキャッシュが揃っていれば組み立てたデータセット、無ければ []。
%
%   シナリオ単位で 8 kHz 生データを **double** で保存するので、中断しても続きから再開できる
%   （書き込み途中のファイルは残さない）。Parallel Computing Toolbox は使えないため、別プロセスで
%   shard を指定して分担する（tools/sa_gen_worker.m）。各プロセスは自分の作業フォルダでモデルを生成する。
%   組み立ては saAssembleDataset（1 kHz 変換、リミット拘束フラグ、train のみの正規化統計、マニフェスト）。
c = sa_config();
if nargin < 1, opts = struct(); end
def = struct('nAx',3,'preset','full','names',{{}},'outDir',c.dataDir,'shard',[1 1],'maxScen',Inf,'cacheOnly',false,'save',true,'replicas',0,'payloadGrid',[1 2 4 6 8]);
f = fieldnames(def);
for i = 1:numel(f)
    if ~isfield(opts,f{i}) || (isempty(opts.(f{i})) && ~strcmp(f{i},'names')), opts.(f{i}) = def.(f{i}); end
end
nAx = opts.nAx;
S = saScenarioSetPayload(nAx, opts.preset, opts.replicas, opts.payloadGrid);
tagPl = '';  if opts.replicas > 0, tagPl = 'pl'; end
if ~isempty(opts.names)
    [tf, loc] = ismember(opts.names, {S.name});
    assert(all(tf), 'genSaDataset:names', '未知のシナリオ名があります');
    S = S(sort(loc));
end
cacheDir = fullfile(opts.outDir,'cache');
if ~isfolder(cacheDir), mkdir(cacheDir); end
cacheFn = @(s) fullfile(cacheDir, sprintf('%s%d_%s.mat', opts.preset, nAx, s.name));

% ---- 1. 未生成のシナリオを（自分の分担ぶんだけ）実行 ----
todo = find(arrayfun(@(s) ~isfile(cacheFn(s)), S));
mine = todo(mod(todo-1, opts.shard(2)) == opts.shard(1)-1);
fprintf('=== %d 軸データ生成 preset=%s: 全 %d 本、未生成 %d 本、shard %d/%d の担当 %d 本 ===\n', ...
    nAx, opts.preset, numel(S), numel(todo), opts.shard(1), opts.shard(2), numel(mine));
if ~isempty(mine)
    workDir = tempname;  mkdir(workDir);                 % Simscape のキャッシュをプロセスごとに分離
    oldDir = cd(workDir);
    cleanWork = onCleanup(@() cleanupWork(oldDir, workDir)); %#ok<NASGU>
    tag = sprintf('%d_%d', opts.shard(1), round(1e6*rand));
    models = struct();                                   % ペイロードごとのモデル（必要になったときに生成）
    nNew = 0;  repoState = saRepoState();
    for k = mine
        if nNew >= opts.maxScen, break; end
        s = S(k);  t0 = tic;
        key = sprintf('p%g', s.payload);  key(key=='.') = '_';
        if ~isfield(models, key), models.(key) = buildModels(nAx, workDir, sprintf('%s_%s', tag, key), s.payload); end
        raw = runSaScenario(s, models.(key));
        e = struct('name',s.name,'seed',s.seed,'q',raw.q,'dq',raw.dq,'tau',raw.tau, ...
            'qref',raw.qref,'dqref',raw.dqref,'q0',raw.q0,'dq0',raw.dq0,'contact8k',raw.contact,'fs',c.fsSim, ...
            'payload',s.payload,'wall',toc(t0),'repo',repoState,'matlab',version); %#ok<NASGU>
        tmp = [cacheFn(s) '.tmp'];  save(tmp, 'e', '-mat');  movefile(tmp, cacheFn(s));   % 書き込み途中のファイルを残さない
        nNew = nNew + 1;
        fprintf('  [%3d/%3d] %-22s %-10s %6.1f s 実行 %6.1f s 経過\n', k, numel(S), s.name, s.type, raw.t(end), toc(t0));
    end
    for fn = fieldnames(models)', for m = {models.(fn{1}).excite, models.(fn{1}).closed}, if bdIsLoaded(m{1}), close_system(m{1},0); end, end, end
end

% ---- 2. すべて揃っていれば組み立て ----
ds = [];
if opts.cacheOnly, return; end
if any(arrayfun(@(s) ~isfile(cacheFn(s)), S))
    fprintf('未生成のシナリオが残っています。同じ引数で再実行すると続きから生成します。\n');
    return;
end
ds = saAssembleDataset(S, @(s) loadCache(cacheFn(s)), nAx, struct('preset',opts.preset));
if opts.save
    fn = fullfile(opts.outDir, sprintf('sa_dataset_%s%s%d.mat', opts.preset, tagPl, nAx));
    if ~isempty(opts.names), fn = fullfile(opts.outDir, sprintf('sa_dataset_%s%s%d_subset.mat', opts.preset, tagPl, nAx)); end
    save(fn, '-struct', 'ds', '-v7.3');
    fprintf('保存: %s（%d シナリオ、%d 遷移）\n', fn, ds.manifest.nScenario, ds.manifest.nTransition);
end
end

function e = loadCache(fn)
C = load(fn);  e = C.e;
if ~isfield(e,'payload'), e.payload = 0; end
end

function m = buildModels(nAx, workDir, tag, payload)
% ペイロードを反映した励振・閉ループのモデル（モデル名はプロセス・ペイロードごとに一意）
nE = sprintf('sa_excite%d_gen%s', nAx, tag);  nC = sprintf('sa_closed%d_gen%s', nAx, tag);
po = struct('mass', payload);
m.excite = buildSaExcitation(nAx, struct('modelName',nE,'slxPath',fullfile(workDir,[nE '.slx']),'payload',po));
m.closed = buildSaClosedLoop(nAx, struct('modelName',nC,'slxPath',fullfile(workDir,[nC '.slx']),'payload',po));
end

function cleanupWork(oldDir, workDir)
cd(oldDir);
if isfolder(workDir), try, rmdir(workDir,'s'); catch, end, end
end
