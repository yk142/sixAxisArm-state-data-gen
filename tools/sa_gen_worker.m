function sa_gen_worker(nAx, preset, shardI, shardN, replicas)
%SA_GEN_WORKER データ生成を分担する作業プロセス用のエントリポイント。
%   別プロセスから実行: matlab -batch "addpath('tools'); sa_gen_worker(3, 'full', 1, 3)"
%   shardI/shardN で担当を分ける（Parallel Computing Toolbox が使えないための代替）。
%   replicas（既定 0）: ペイロードを変えた複製の組数（issue #65、saScenarioSetPayload）。
%   生成だけ行い（cacheOnly）、組み立ては全員の生成後に genSaDataset を 1 回呼んで行う。
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
if ischar(nAx), nAx = str2double(nAx); end
if ischar(shardI), shardI = str2double(shardI); end
if ischar(shardN), shardN = str2double(shardN); end
if nargin < 5, replicas = 0; end
if ischar(replicas), replicas = str2double(replicas); end
genSaDataset(struct('nAx',nAx,'preset',preset,'shard',[shardI shardN],'cacheOnly',true,'replicas',replicas));
end
