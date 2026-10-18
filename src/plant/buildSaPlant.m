function mdl = buildSaPlant(nAx, opts)
%BUILDSAPLANT 先頭 nAx 軸を可動、残りを 0 rad で固定した C8-A901S の Simscape プラントを生成する。
%   mdl = BUILDSAPLANT(nAx)         nAx = 3（J1〜J3）または 6。data/models/sa_plant<nAx>.slx に保存
%   mdl = BUILDSAPLANT(nAx, opts)   構造体 opts で設定を上書き
%     opts.modelName  (既定 'sa_plant<nAx>')
%     opts.slxPath    (既定 <dataDir>/models/<modelName>.slx)
%     opts.solver     'ode15s'（既定。可変刻み、MaxStep = 1/8 kHz、RelTol 1e-6、AbsTol 1e-8）または 'ode14x'（固定刻み 8 kHz）
%     opts.filterTc   Solver Configuration の FilteringTimeConstant [s]（既定 1e-6。Simscape の既定は 1e-3）。
%                     RMI（armature）の 1-D/3-D 接続の自動フィルタの時定数で、小さいほど応答の遅れが減る。
%                     ode14x では 5e-4 以下で不安定になる（docs/phase1_report.md）。ode15s では 1e-6 まで安定
%
%   共通 I/F: Inport 'tau' [nAx] (N*m) → Outport 'q' [nAx] (rad), 'dq' [nAx] (rad/s)
%   可動関節のブロック名は 'joint1'..'joint<nAx>'（親資産 addC8* の前提）。
%   方式（docs/phase0_report.md）: 親の writeC8URDF の joint(nAx+1..6) を fixed に後処理 → smimport
%   → ode15s（armature の RMI が DAE を作るため陰的ソルバが必要）。親資産は変更しない。
%   RMI の自動フィルタ（時定数 1 ms）が応答を遅らせるため filterTc を小さくする（docs/phase1_report.md）。
c = sa_config();
if nargin < 2, opts = struct(); end
if ~isfield(opts,'modelName') || isempty(opts.modelName), opts.modelName = sprintf('sa_plant%d',nAx); end
if ~isfield(opts,'slxPath') || isempty(opts.slxPath)
    opts.slxPath = fullfile(c.dataDir,'models',[opts.modelName '.slx']);
end
if ~isfield(opts,'solver') || isempty(opts.solver), opts.solver = 'ode15s'; end
if ~isfield(opts,'filterTc') || isempty(opts.filterTc), opts.filterTc = 1e-6; end
assert(any(nAx == [3 6]), 'buildSaPlant:nAx', 'nAx は 3 または 6');
slxDir = fileparts(opts.slxPath);
if ~isfolder(slxDir), mkdir(slxDir); end

p = c8_params(c.meshDir);
if isfield(opts,'payload'), p = saApplyPayload(p, opts.payload); end
urdf6 = [tempname '.urdf'];  urdf = [tempname '.urdf'];
cleanup = onCleanup(@() cellfun(@(f) delete_if_exists(f), {urdf6, urdf}));
writeC8URDF(p, urdf6);
txt = fileread(urdf6);
for i = nAx+1:6
    blk = regexp(txt, sprintf('<joint name="joint%d" type="revolute">.*?</joint>', i), 'match', 'once');
    assert(~isempty(blk), 'buildSaPlant:noJoint', 'URDF に joint%d が見つかりません', i);
    nb = strrep(blk, 'type="revolute"', 'type="fixed"');
    nb = regexprep(nb, '\s*<axis[^>]*/>', '');
    nb = regexprep(nb, '\s*<limit[^>]*/>', '');
    txt = strrep(txt, blk, nb);
end
fid = fopen(urdf,'w');  assert(fid > 0, 'buildSaPlant:open', '書き込めません: %s', urdf);
fwrite(fid, txt);  fclose(fid);

mdl = opts.modelName;
if bdIsLoaded(mdl), close_system(mdl,0); end
smimport(urdf, 'ModelName', mdl);
mc = find_system(mdl,'MaskType','Mechanism Configuration');
set_param(mc{1}, 'GravityVector', sprintf('[%.9g %.9g %.9g]', p.gravity));
set_param(mdl,'SolverType','Fixed-step','Solver','ode14x', ...
    'FixedStep',num2str(1/c.fsSim,'%.12g'),'StopTime','0.5','SimscapeLogType','none');
try, set_param(mdl,'SimMechanicsOpenEditorOnUpdate','off'); catch, end
if strcmp(opts.solver,'ode15s')
    set_param(mdl,'SolverType','Variable-step','Solver','ode15s','MaxStep',num2str(1/c.fsSim,'%.12g'), ...
        'RelTol','1e-6','AbsTol','1e-8');
end
scb = find_system(mdl,'MaskType','Solver Configuration');
set_param(scb{1},'FilteringTimeConstant',num2str(opts.filterTc,'%.12g'));
pn = p;  pn.nJoints = nAx;
addC8Actuation(mdl, pn);  addC8Armature(mdl, pn);  addC8Friction(mdl, pn);
save_system(mdl, opts.slxPath);
fprintf('buildSaPlant: 保存 %s (revolute=%d, step=%.4g s)\n', opts.slxPath, ...
    numel(find_system(mdl,'MaskType','Revolute Joint')), 1/c.fsSim);
end

function delete_if_exists(f)
if exist(f,'file'), delete(f); end
end
