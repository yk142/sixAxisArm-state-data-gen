function sa_export_closed_eval(outFile, nAx)
%SA_EXPORT_CLOSED_EVAL 閉ループ評価用データを HDF5 に書き出す（issue #45）。
%   sa_export_closed_eval('data/export/closed_eval3.h5')       % 3 軸（既定）
%   sa_export_closed_eval('data/export/closed_eval6.h5', 6)    % 6 軸（issue #55。新規シードは 175000 + i）
%   対象:
%     ptp       ベンチマーク PTP 12 本（真値はキャッシュの Simscape 結果）
%     nearlimit 学習に使っていない保持シナリオ: test の既存 2 本＋新規 12 本（シード = seedBase + 170000 + i、Simscape を新たに実行）
%   各シナリオのグループ: ref [N x (1+3·nAx)] = [t, q_ref, dq_ref, tau_ff]、tau_d [N x nAx]、truth_q / truth_dq / truth_tau [N x nAx]（8 kHz）。
%   実行: matlab -batch "addpath('tools'); sa_export_closed_eval('data/export/closed_eval3.h5')"
repo = fileparts(fileparts(mfilename('fullpath')));
if ~startsWith(outFile, filesep), outFile = fullfile(pwd, outFile); end      % 作業フォルダへ移るので絶対パスにする
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
if nargin < 2, nAx = 3; end
c = sa_config();  seedOff = 170000 + 5000*(nAx == 6);
S = saScenarioSet(nAx, 'full');
ptp = S(strcmp({S.type},'ptp'));
near = S(strcmp({S.pattern},'nearlimit'));
old = near(strcmp({near.split},'test'));
new = near;
for i = 1:numel(new)
    new(i).seed = c.seedBase + seedOff + i;
    new(i).name = ['cl_' near(i).name];
    new(i).split = 'eval';
end
if exist(outFile,'file'), delete(outFile); end
workDir = tempname;  mkdir(workDir);  oldDir = cd(workDir);
cleanWork = onCleanup(@() cd(oldDir)); %#ok<NASGU>
nC = sprintf('sa_closed%d_cleval', nAx);
mC = buildSaClosedLoop(nAx, struct('modelName',nC,'slxPath',fullfile(workDir,[nC '.slx'])));
models = struct('closed',mC);
list = [ptp, old, new];
for i = 1:numel(list)
    s = list(i);  t0 = tic;
    [REF, tauD] = saClosedInputs(s);
    if strcmp(s.type,'ptp') || any(strcmp(s.name, {old.name}))
        cf = fullfile(c.dataDir,'cache', sprintf('full%d_%s.mat', nAx, s.name));
        e = load(cf, 'e');  e = e.e;  tq = e.q;  tdq = e.dq;  ttau = e.tau;
        assert(max(abs(e.qref - REF(:,2:1+nAx)), [], "all") < 1e-9, '参照の再構成がキャッシュと一致しません: %s', s.name);
    else
        raw = runSaScenario(s, models);
        tq = raw.q;  tdq = raw.dq;  ttau = raw.tau;
        assert(max(abs(raw.qref - REF(:,2:1+nAx)), [], 'all') < 1e-9, '参照の再構成が実行と一致しません: %s', s.name);
    end
    g = ['/' s.name];
    h5w = @(k, v) h5create_write(outFile, [g '/' k], v);
    h5w('ref', REF);  h5w('tau_d', tauD);  h5w('truth_q', tq);  h5w('truth_dq', tdq);  h5w('truth_tau', ttau);
    h5writeatt(outFile, g, 'type', s.type);  h5writeatt(outFile, g, 'pattern', s.pattern);  h5writeatt(outFile, g, 'split', s.split);
    fprintf('%2d/%d %s (%s) %.0f s\n', i, numel(list), s.name, s.split, toc(t0));
end
% 制御器の定数
sp = saParams(nAx);
h5create_write(outFile, '/controller/Mmean', sp.Mmean(:)');  h5create_write(outFile, '/controller/tauPeak', sp.tauPeak(:)');
h5create_write(outFile, '/controller/wn', 60);  h5create_write(outFile, '/controller/zeta', 1);
end

function h5create_write(f, name, v)
h5create(f, name, size(v), 'Datatype', 'double');
h5write(f, name, double(v));
end
