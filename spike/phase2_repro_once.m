function phase2_repro_once(outFile)
%PHASE2_REPRO_ONCE 別プロセスでの再現性確認用: 6 軸モデルで固定シナリオを格子出力で実行して保存する。
cd(fileparts(fileparts(mfilename('fullpath'))));  addpath('config','spike');  addpath(genpath('src'));
c = sa_config();  mdl = 'sa_plant6';  load_system(fullfile(c.dataDir,'models',[mdl '.slx']));
S = phase2_scenario(6, 0.5, 5);  o = phase2_run(mdl, S, 'grid');  save(outFile, 'o');
end
