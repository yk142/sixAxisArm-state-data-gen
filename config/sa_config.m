function c = sa_config()
%SA_CONFIG 6 軸データ生成の設定（親資産パス、サンプリング）を返す。
%   親資産 robotArmSurrogate-matlab は参照専用。パスは環境変数 SA_PARENT_DIR で上書きできる。
c.parentDir = getenv('SA_PARENT_DIR');
if isempty(c.parentDir)
    c.parentDir = fullfile(getenv('HOME'),'matlab-projects','robotArmSurrogate-matlab');
end
c.meshDir = fullfile(c.parentDir,'urdf','mesh');
c.fsSim   = 8000;
c.fsData  = 1000;
c.seedBase = 20261011;
root = fileparts(fileparts(mfilename('fullpath')));
c.rootDir = root;
c.dataDir = fullfile(root,'data');
addpath(fullfile(c.parentDir,'config'));
addpath(genpath(fullfile(c.parentDir,'src')));
end
