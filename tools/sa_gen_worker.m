function sa_gen_worker(nAx, preset, shardI, shardN, replicas, pulse1, pulseOther, elastic)
%SA_GEN_WORKER データ生成を分担する作業プロセス用のエントリポイント。
%   別プロセスから実行: matlab -batch "addpath('tools'); sa_gen_worker(3, 'full', 1, 3)"
%   shardI/shardN で担当を分ける（Parallel Computing Toolbox が使えないための代替）。
%   replicas（既定 0）: ペイロードを変えた複製の組数（issue #65、saScenarioSetPayload）。
%   pulse1 / pulseOther（既定 0）: 高トルクのパルス励振の本数（軸 1 / 他軸。issue #67、saScenarioSetPulse）。
%   生成だけ行い（cacheOnly）、組み立ては全員の生成後に genSaDataset を 1 回呼んで行う。
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
if ischar(nAx), nAx = str2double(nAx); end
if ischar(shardI), shardI = str2double(shardI); end
if ischar(shardN), shardN = str2double(shardN); end
if nargin < 5, replicas = 0; end
if ischar(replicas), replicas = str2double(replicas); end
if nargin < 6, pulse1 = 0; end
if nargin < 7, pulseOther = 0; end
if ischar(pulse1), pulse1 = str2double(pulse1); end
if ischar(pulseOther), pulseOther = str2double(pulseOther); end
if nargin < 8, elastic = []; end                 % 1 を渡すと既定の関節弾性（f = 20 Hz。issue #73）
if isequal(elastic, 0) || isequal(elastic, '0'), elastic = []; elseif isequal(elastic, 1) || isequal(elastic, '1'), elastic = struct(); end
genSaDataset(struct('nAx',nAx,'preset',preset,'shard',[shardI shardN],'cacheOnly',true,'replicas',replicas,'pulse',[pulse1 pulseOther],'elastic',elastic));
end
