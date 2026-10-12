function sa_repro_worker(nAx, preset, mode, shardI, shardN)
%SA_REPRO_WORKER 再現性確認用: 既存キャッシュとは別フォルダ（data/repro）へ、キャッシュなしで再生成する。
%   matlab -batch "addpath('tools'); sa_repro_worker(3,'full','sample',1,2)"   mode = 'sample' | 'all'
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
if ischar(nAx), nAx = str2double(nAx); end
if ischar(shardI), shardI = str2double(shardI); end
if ischar(shardN), shardN = str2double(shardN); end
S = saScenarioSet(nAx, preset);
names = saReproSample(S, mode);
c = sa_config();
genSaDataset(struct('nAx',nAx,'preset',preset,'names',{names},'outDir',fullfile(c.dataDir,'repro'), ...
    'shard',[shardI shardN],'cacheOnly',true));
end
