function sa_export_closed_um(outFile, names, tol, unmodeled, ptpExtra)
%SA_EXPORT_CLOSED_UM 閉ループのシナリオ（hold / ptp）を Simscape で実行して HDF5 に書く（issue #99、同定・学習用）。
%   sa_export_closed_um('data/export/id_elum_h1.h5', {'hold_p1_001',...}, struct('RelTol',1e-7,'AbsTol',1e-9), struct())
%   sa_export_closed_um('data/export/id_elum_p1.h5', {}, tol, struct(), [0.3 1; 0.5 1])     追加の PTP（速度倍率・番号）
%   names: saScenarioSetPayload(6,'full',0,[]) の閉ループシナリオ名。ptpExtra [n x 2]: [speedScale, k] の PTP を新しいシードで作る
%     （ベンチマーク ptp_v20_001 を雛形に、名前 ptpid_v<速度×100>_<k>、seed = 9000000 + 100·速度×100 + k、split 'idtrain'）
%   tol: 許容誤差（RelTol / AbsTol / MaxStep）、unmodeled: addSaUnmodeled の um（[] で効果なし）
%   形式は sa_export_ptp_el と同じに加えて truth_tau（実印加トルク、8 kHz）を書く。payload 0 kg、PD wn 60・ζ 1
repo = fileparts(fileparts(mfilename('fullpath')));
if ~startsWith(outFile, filesep), outFile = fullfile(pwd, outFile); end
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
if nargin < 5, ptpExtra = []; end
S0 = saScenarioSetPayload(6,'full',0,[]);
S = S0(ismember({S0.name}, names));
tpl = S0(strcmp({S0.name},'ptp_v20_001'));
for r = 1:size(ptpExtra,1)
    s = tpl;  v = round(100*ptpExtra(r,1));
    s.name = sprintf('ptpid_v%d_%d', v, ptpExtra(r,2));  s.speedScale = ptpExtra(r,1);
    s.seed = 9000000 + 100*v + ptpExtra(r,2);  s.split = 'idtrain';  s.pattern = sprintf('ptp_v%d', v);
    S(end+1) = s; %#ok<AGROW>
end
if exist(outFile,'file'), delete(outFile); end
workDir = tempname;  mkdir(workDir);  oldDir = cd(workDir);  cl = onCleanup(@() cd(oldDir)); %#ok<NASGU>
oc = struct('modelName','sa_closed6_um','slxPath',fullfile(workDir,'sa_closed6_um.slx'),'payload',struct('mass',0),'elastic',struct());
if ~isempty(unmodeled), oc.unmodeled = unmodeled; end
mdl = buildSaClosedLoop(6, oc);
c = sa_config();  rt = '1e-6'; at = '1e-8'; ms = num2str(1/c.fsSim,'%.12g');
if isfield(tol,'RelTol'), rt = num2str(tol.RelTol,'%.12g'); end
if isfield(tol,'AbsTol'), at = num2str(tol.AbsTol,'%.12g'); end
if isfield(tol,'MaxStep'), ms = num2str(tol.MaxStep,'%.12g'); end
set_param(mdl,'RelTol',rt,'AbsTol',at,'MaxStep',ms);
fprintf('Simscape（unmodeled=%d）: RelTol %s AbsTol %s、%d 本\n', ~isempty(unmodeled), rt, at, numel(S));
for i = 1:numel(S)
    s = S(i);  t0 = tic;  g = ['/' s.name];
    [REF, tauD] = saClosedInputs(s);
    o = simSaClosed(mdl, REF(:,1), REF, tauD, REF(1,2:7)');
    w = @(k, v) wr(outFile, [g '/' k], v);
    w('ref', REF);  w('tau_d', tauD);  w('truth_q', o.q);  w('truth_dq', o.dq);  w('truth_qm', o.qm);  w('truth_dqm', o.dqm);  w('truth_tau', o.tau);
    h5writeatt(outFile, g, 'type', s.type);  h5writeatt(outFile, g, 'pattern', s.pattern);  h5writeatt(outFile, g, 'split', s.split);
    h5writeatt(outFile, g, 'sec', toc(t0));
    fprintf('%2d/%d %s (sim) %.1f s\n', i, numel(S), s.name, toc(t0));
end
close_system(mdl, 0);
sp = saParams(6);
wr(outFile, '/controller/Mmean', sp.Mmean(:)');  wr(outFile, '/controller/tauPeak', sp.tauPeak(:)');
wr(outFile, '/controller/wn', 60);  wr(outFile, '/controller/zeta', 1);
end

function wr(f, name, v)
h5create(f, name, size(v), 'Datatype', 'double');
h5write(f, name, double(v));
end
