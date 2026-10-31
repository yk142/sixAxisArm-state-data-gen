function sa_export_ptp_el(outFile, tol, names)
%SA_EXPORT_PTP_EL 弾性真値の PTP（ペイロード 0 kg、12 本）を HDF5 に書く（issue #93、位置決め精度の評価用）。
%   sa_export_ptp_el('data/export/ptp_el.h5')                      キャッシュ（data/cache/fullel6_*.mat）から書く（Simscape は再実行しない）
%   sa_export_ptp_el('data/export/ptp_el_tight.h5', struct('RelTol',1e-9,'AbsTol',1e-11))   許容誤差を変えて Simscape を再実行して書く
%   tol.RelTol / tol.AbsTol / tol.MaxStep（省略で既定 1e-6 / 1e-8 / 1/8 kHz）。names で本数を絞れる（例 {'ptp_v20_001'}）
%   形式は sa_export_eval_el と同じ（ref、tau_d、truth_q / dq / qm / dqm を 8 kHz、controller）。属性に許容誤差と計算時間
repo = fileparts(fileparts(mfilename('fullpath')));
if ~startsWith(outFile, filesep), outFile = fullfile(pwd, outFile); end
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
if nargin < 2, tol = []; end
S = saScenarioSetPayload(6,'full',0,[]);  S = S(strcmp({S.type},'ptp'));
if nargin >= 3 && ~isempty(names), S = S(ismember({S.name}, names)); end
if exist(outFile,'file'), delete(outFile); end
mdl = '';
if ~isempty(tol)
    workDir = tempname;  mkdir(workDir);  oldDir = cd(workDir);  cl = onCleanup(@() cd(oldDir)); %#ok<NASGU>
    oc = struct('modelName','sa_closed6_ptptol','slxPath',fullfile(workDir,'sa_closed6_ptptol.slx'),'payload',struct('mass',0),'elastic',struct());
    mdl = buildSaClosedLoop(6, oc);
    c = sa_config();
    rt = '1e-6'; at = '1e-8'; ms = num2str(1/c.fsSim,'%.12g');
    if isfield(tol,'RelTol'), rt = num2str(tol.RelTol,'%.12g'); end
    if isfield(tol,'AbsTol'), at = num2str(tol.AbsTol,'%.12g'); end
    if isfield(tol,'MaxStep'), ms = num2str(tol.MaxStep,'%.12g'); end
    set_param(mdl,'RelTol',rt,'AbsTol',at,'MaxStep',ms);
    fprintf('Simscape 再実行: RelTol %s AbsTol %s MaxStep %s（solver %s）\n', get_param(mdl,'RelTol'), get_param(mdl,'AbsTol'), get_param(mdl,'MaxStep'), get_param(mdl,'Solver'));
end
for i = 1:numel(S)
    s = S(i);  t0 = tic;  g = ['/' s.name];
    [REF, tauD] = saClosedInputs(s);
    if isempty(mdl)
        x = load(fullfile(repo,'data/cache',sprintf('fullel6_%s.mat', s.name)));  e = x.e;
        q = e.q; dq = e.dq; qm = e.qm; dqm = e.dqm;  src = 'cache';
    else
        tVec = REF(:,1);
        o = simSaClosed(mdl, tVec, REF, tauD, REF(1,2:7)');
        q = o.q; dq = o.dq; qm = o.qm; dqm = o.dqm;  src = 'sim';
    end
    w = @(k, v) wr(outFile, [g '/' k], v);
    w('ref', REF);  w('tau_d', tauD);  w('truth_q', q);  w('truth_dq', dq);  w('truth_qm', qm);  w('truth_dqm', dqm);
    h5writeatt(outFile, g, 'type', s.type);  h5writeatt(outFile, g, 'pattern', s.pattern);  h5writeatt(outFile, g, 'split', s.split);
    h5writeatt(outFile, g, 'source', src);  h5writeatt(outFile, g, 'sec', toc(t0));
    fprintf('%2d/%d %s (%s) %.1f s\n', i, numel(S), s.name, src, toc(t0));
end
if ~isempty(mdl), close_system(mdl, 0); end
sp = saParams(6);
wr(outFile, '/controller/Mmean', sp.Mmean(:)');  wr(outFile, '/controller/tauPeak', sp.tauPeak(:)');
wr(outFile, '/controller/wn', 60);  wr(outFile, '/controller/zeta', 1);
end

function wr(f, name, v)
h5create(f, name, size(v), 'Datatype', 'double');
h5write(f, name, double(v));
end
