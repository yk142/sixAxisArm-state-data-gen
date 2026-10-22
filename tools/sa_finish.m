function ds = sa_finish(nAx, preset, replicas, pulse1, pulseOther, elastic)
%SA_FINISH 全シナリオのキャッシュが揃ったデータセットを組み立てて検証する。
%   matlab -batch "addpath('tools'); sa_finish(3, 'full')"
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
if ischar(nAx), nAx = str2double(nAx); end
if nargin < 3, replicas = 0; end
if ischar(replicas), replicas = str2double(replicas); end
if nargin < 4, pulse1 = 0; end
if nargin < 5, pulseOther = 0; end
if ischar(pulse1), pulse1 = str2double(pulse1); end
if ischar(pulseOther), pulseOther = str2double(pulseOther); end
if nargin < 6, elastic = []; end
if isequal(elastic, 0) || isequal(elastic, '0'), elastic = []; elseif isequal(elastic, 1) || isequal(elastic, '1'), elastic = struct(); end
ds = genSaDataset(struct('nAx',nAx,'preset',preset,'replicas',replicas,'pulse',[pulse1 pulseOther],'elastic',elastic));
assert(~isempty(ds), 'sa_finish:incomplete', '未生成のシナリオが残っています');
r = validateSaData(ds);
assert(r.ok, 'sa_finish:validate', '検証に不合格の項目があります');
end
