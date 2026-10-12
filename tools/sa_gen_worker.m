function sa_gen_worker(nAx, preset, shardI, shardN)
%SA_GEN_WORKER データ生成を分担する作業プロセス用のエントリポイント。
%   別プロセスから実行: matlab -batch "addpath('tools'); sa_gen_worker(3, 'full', 1, 3)"
%   shardI/shardN で担当を分ける（Parallel Computing Toolbox が使えないための代替）。
%   生成だけ行い（cacheOnly）、組み立ては全員の生成後に genSaDataset を 1 回呼んで行う。
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
if ischar(nAx), nAx = str2double(nAx); end
if ischar(shardI), shardI = str2double(shardI); end
if ischar(shardN), shardN = str2double(shardN); end
genSaDataset(struct('nAx',nAx,'preset',preset,'shard',[shardI shardN],'cacheOnly',true));
end
