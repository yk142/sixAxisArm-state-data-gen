function sa_export_closed_contact(outFile)
%SA_EXPORT_CLOSED_CONTACT リミット押し付けの閉ループ評価データを HDF5 に書き出す（issue #47）。
%   sa_export_closed_contact('data/export/closed_contact3.h5')
%   nearlimit 12 本の姿勢・外乱を流用し、目標軸の参照を 1.0〜2.5 s でリミットの pushDepth 外側へ移す（saClosedInputs）。
%   pushDepth は 2° と 5°（同じ軸・側の 001 / 002）。外乱は nearlimit と同じ BLN。シード = seedBase + 180000 + i、split = 'eval'。
%   形式は sa_export_closed_eval と同じ（ref, tau_d, truth_q, truth_dq, truth_tau, 属性、/controller）。
%   実行: matlab -batch "addpath('tools'); sa_export_closed_contact('data/export/closed_contact3.h5')"
repo = fileparts(fileparts(mfilename('fullpath')));
if ~startsWith(outFile, filesep), outFile = fullfile(pwd, outFile); end
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
c = sa_config();  nAx = 3;  d2r = pi/180;
S = saScenarioSet(nAx, 'full');
near = S(strcmp({S.pattern},'nearlimit'));
new = near;
for i = 1:numel(new)
    new(i).seed = c.seedBase + 180000 + i;
    new(i).name = ['cp_' near(i).name];
    new(i).split = 'eval';
end
% 同じ軸・側で 2 本: 1 周目 = 2°、2 周目 = 5°（nearlimit は 軸 3 × 両側 2 = 6 本で 1 周）
for i = 1:numel(new), new(i).pushDepth = (2 + 3*(i > nAx*2))*d2r; end
if exist(outFile,'file'), delete(outFile); end
workDir = tempname;  mkdir(workDir);  oldDir = cd(workDir);
cleanWork = onCleanup(@() cd(oldDir)); %#ok<NASGU>
nC = 'sa_closed3_ctact';
mC = buildSaClosedLoop(nAx, struct('modelName',nC,'slxPath',fullfile(workDir,[nC '.slx'])));
sp = saParams(nAx);
for i = 1:numel(new)
    s = new(i);  t0 = tic;
    [REF, tauD] = saClosedInputs(s);
    o = simSaClosed(mC, REF(:,1), REF, tauD, REF(1,2:1+nAx)');       % runSaScenario の hold と同じ呼び出し
    g = ['/' s.name];
    h5w = @(k, v) h5create_write(outFile, [g '/' k], v);
    h5w('ref', REF);  h5w('tau_d', tauD);  h5w('truth_q', o.q);  h5w('truth_dq', o.dq);  h5w('truth_tau', o.tau);
    h5writeatt(outFile, g, 'type', s.type);  h5writeatt(outFile, g, 'pattern', s.pattern);  h5writeatt(outFile, g, 'split', s.split);
    h5writeatt(outFile, g, 'pushDepthDeg', s.pushDepth/d2r);
    nearLim = o.q <= sp.qMin + 0.1*d2r | o.q >= sp.qMax - 0.1*d2r;
    fprintf('%2d/%d %s depth %g deg, 接触割合 %.3f, %.0f s\n', i, numel(new), s.name, s.pushDepth/d2r, mean(any(nearLim,2)), toc(t0));
end
h5create_write(outFile, '/controller/Mmean', sp.Mmean(:)');  h5create_write(outFile, '/controller/tauPeak', sp.tauPeak(:)');
h5create_write(outFile, '/controller/wn', 60);  h5create_write(outFile, '/controller/zeta', 1);
end

function h5create_write(f, name, v)
h5create(f, name, size(v), 'Datatype', 'double');
h5write(f, name, double(v));
end
