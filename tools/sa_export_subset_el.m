function sa_export_subset_el(outFile)
%SA_EXPORT_SUBSET_EL キャッシュが揃っている弾性つきシナリオだけでデータセットを組み立て、HDF5 に書き出す（issue #73。縮小版）。
%   全シナリオ表（複製 1 組、パルス 120/12）のうちキャッシュのあるものを使う。validateSaData は剛体の解析と比べる検査なので行わない。
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
if nargin < 1, outFile = fullfile(repo,'data/export/sa_flat_fullplpu6el.h5'); end
S = [saScenarioSetPayload(6,'full',1,[1 2 4 6 8]), saScenarioSetPulse(6,120,12,[0 1 2 4 6 8])];
cf = @(s) fullfile(repo,'data/cache',sprintf('fullel6_%s.mat',s.name));
S = S(arrayfun(@(s) isfile(cf(s)), S));
fprintf('キャッシュのあるシナリオ %d 本\n', numel(S));
ds = saAssembleDataset(S, @(s) getfield(load(cf(s)),'e'), 6, struct('preset','fullplpuel')); %#ok<GFLD>
fprintf('遷移 %d\n', ds.manifest.nTransition);
saExportFlat(ds, outFile, struct('check',false));
fprintf('書き出し: %s\n', outFile);
end
