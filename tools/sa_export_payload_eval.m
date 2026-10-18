function sa_export_payload_eval(outFile, payloads, wns, zetas)
%SA_EXPORT_PAYLOAD_EVAL 手先ペイロードと制御ゲインを変えた 6 軸の閉ループステップ応答（Simscape の真値）を HDF5 に書く（issue #61）。
%   sa_export_payload_eval('data/export/payload_eval6.h5')
%   既定: ペイロード [0 1 3 5 8] kg（arm6 座標系の [0 0 0.05] m）、wn = [30 60 100 150]、ζ = [0.5 1 2]、各 6 軸
%   シナリオ: 姿勢 P0（saScenarioSet の p1）で静止 → t = 0.6 s に 1 軸だけ 2° のステップ（参照）。フィードフォワードはペイロード込みの RNEA（重力たわみを除いてゲインの効果を見る）。
%   制御器: PD（Kp = Mmean·wn²、Kd = 2ζ·Mmean·wn、公称の Mmean）、軸ごとに ±tauPeak。ペイロードごとにモデルを 1 回作り、ゲインは set_param で切り替える。
%   グループ /pl<kg>_wn<wn>_z<ζ×100>_j<axis>: ref [N x 19]、tau_d、truth_q/dq/tau（8 kHz）、Kp・Kd、属性
%   実行: matlab -batch "addpath('tools'); sa_export_payload_eval('data/export/payload_eval6.h5')"
repo = fileparts(fileparts(mfilename('fullpath')));
if ~startsWith(outFile, filesep), outFile = fullfile(pwd, outFile); end
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
if nargin < 2 || isempty(payloads), payloads = [0 1 3 5 8]; end
if nargin < 3 || isempty(wns), wns = [30 60 100 150]; end
if nargin < 4 || isempty(zetas), zetas = [0.5 1 2]; end
nAx = 6;  c = sa_config();  dt = 1/c.fsSim;  sp = saParams(nAx);
P0 = [0.0 -0.5 0.5 0.4 0.8 -0.4];  step = deg2rad(2);  tStep = 0.6;  tEnd = 1.8;
tVec = (0:dt:tEnd)';  nT = numel(tVec);
if exist(outFile,'file'), delete(outFile); end
workDir = tempname;  mkdir(workDir);  oldDir = cd(workDir);
cleanWork = onCleanup(@() cd(oldDir)); %#ok<NASGU>
nRun = numel(payloads)*numel(wns)*numel(zetas)*nAx;  k = 0;  t0 = tic;
for pm = payloads
    REF = cell(1, nAx);                       % フィードフォワードはペイロード込みの RNEA（ゲインには依存しない）
    for a = 1:nAx
        qr = repmat(P0, nT, 1);  qr(tVec >= tStep, a) = P0(a) + step;
        REF{a} = [tVec, qr, zeros(nT, nAx), saFeedforward(tVec, qr, zeros(nT,nAx), zeros(nT,nAx), struct('mass',pm))];
    end
    nm = sprintf('sa_closed6_pl%d', pm);
    mdl = buildSaClosedLoop(nAx, struct('modelName',nm,'slxPath',fullfile(workDir,[nm '.slx']),'payload',struct('mass',pm)));
    for wn = wns
        for z = zetas
            Kp = sp.Mmean*wn^2;  Kd = 2*z*sp.Mmean*wn;
            set_param([mdl '/Kp'],'Gain',mat2str(Kp,12));  set_param([mdl '/Kd'],'Gain',mat2str(Kd,12));
            for a = 1:nAx
                o = simSaClosed(mdl, tVec, REF{a}, [], P0');
                g = sprintf('/pl%d_wn%d_z%d_j%d', pm, wn, round(100*z), a);
                w = @(key, v) h5c(outFile, [g '/' key], v);
                w('ref', REF{a});  w('tau_d', zeros(nT, nAx));  w('truth_q', o.q);  w('truth_dq', o.dq);  w('truth_tau', o.tau);
                w('Kp', Kp);  w('Kd', Kd);  w('P0', P0);
                h5writeatt(outFile, g, 'type', 'step');  h5writeatt(outFile, g, 'pattern', 'step');  h5writeatt(outFile, g, 'split', 'eval');
                h5writeatt(outFile, g, 'payload', pm);  h5writeatt(outFile, g, 'wn', wn);  h5writeatt(outFile, g, 'zeta', z);  h5writeatt(outFile, g, 'axis', a);
                h5writeatt(outFile, g, 'step', step);  h5writeatt(outFile, g, 'tStep', tStep);
                k = k + 1;
            end
            fprintf('%3d/%d payload %d kg wn %d zeta %.1f  (%.0f s)\n', k, nRun, pm, wn, z, toc(t0));
        end
    end
    close_system(mdl, 0);
end
h5c(outFile, '/controller/tauPeak', sp.tauPeak(:)');
end

function h5c(f, name, v)
h5create(f, name, size(v), 'Datatype', 'double');  h5write(f, name, double(v));
end
