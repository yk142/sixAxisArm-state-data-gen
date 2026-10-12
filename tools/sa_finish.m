function ds = sa_finish(nAx, preset)
%SA_FINISH 全シナリオのキャッシュが揃ったデータセットを組み立てて検証する。
%   matlab -batch "addpath('tools'); sa_finish(3, 'full')"
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
if ischar(nAx), nAx = str2double(nAx); end
ds = genSaDataset(struct('nAx',nAx,'preset',preset));
assert(~isempty(ds), 'sa_finish:incomplete', '未生成のシナリオが残っています');
r = validateSaData(ds);
assert(r.ok, 'sa_finish:validate', '検証に不合格の項目があります');
end
