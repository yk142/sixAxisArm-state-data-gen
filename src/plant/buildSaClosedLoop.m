function mdl = buildSaClosedLoop(nAx, opts)
%BUILDSACLOSEDLOOP PD＋フィードフォワード閉ループモデル（軸ごとのトルク飽和付き）を生成する。
%   mdl = BUILDSACLOSEDLOOP(nAx)         nAx = 3 または 6。data/models/sa_closed<nAx>.slx に保存
%   mdl = BUILDSACLOSEDLOOP(nAx, opts)   opts は buildSaPlant のものに加え:
%       opts.modelName  既定 'sa_closed<nAx>'
%       opts.wn         PD の固有角周波数 [rad/s]（スカラまたは [1 x nAx]、既定 60）、opts.zeta 減衰比（既定 1）
%   opts.elastic（issue #71）を渡すと関節弾性つき。PD のフィードバックは el.fb（既定 'motor' = モータ側）。Outport 4, 5 = 'qm', 'dqm'（モータ側）
%   I/F:
%     Inport 'ref' [4·nAx] = [q_ref, dq_ref, tau_ff, tau_dist]
%     Outport 'q', 'dq', 'tau'（飽和後の実印加トルク）
%     τ = sat( tau_ff + tau_dist + Kp·(q_ref − q) + Kd·(dq_ref − dq) )、軸ごとに ±tauPeak
%   Kp = Mmean·wn²、Kd = 2·ζ·Mmean·wn（Mmean は saParams の質量行列の対角の平均）。
%   PD は連続形で、ソルバの刻みで評価される。PTP 追従（tau_dist=0）と、姿勢保持＋外乱（tau_dist≠0）に使う。
if nargin < 2, opts = struct(); end
if ~isfield(opts,'modelName') || isempty(opts.modelName), opts.modelName = sprintf('sa_closed%d',nAx); end
wn = 60;   if isfield(opts,'wn') && ~isempty(opts.wn), wn = opts.wn; end
zeta = 1;  if isfield(opts,'zeta') && ~isempty(opts.zeta), zeta = opts.zeta; end
for f = {'wn','zeta'}, if isfield(opts,f{1}), opts = rmfield(opts,f{1}); end, end
mdl = buildSaPlant(nAx, opts);
sp = saParams(nAx);
wn = wn.*ones(1,nAx);

set_param([mdl '/tau'],'Name','ref');
set_param([mdl '/ref'],'PortDimensions',num2str(4*nAx));
delete_line(mdl,'ref/1','tau_demux/1');
add_block('simulink/Signal Routing/Demux',[mdl '/refDemux'],'Outputs',mat2str(nAx*ones(1,4)));
add_block('simulink/Math Operations/Sum',[mdl '/eq'],'Inputs','+-');
add_block('simulink/Math Operations/Sum',[mdl '/edq'],'Inputs','+-');
add_block('simulink/Math Operations/Gain',[mdl '/Kp'],'Gain',mat2str(sp.Mmean.*wn.^2,12),'Multiplication','Element-wise(K.*u)');
add_block('simulink/Math Operations/Gain',[mdl '/Kd'],'Gain',mat2str(2*zeta*sp.Mmean.*wn,12),'Multiplication','Element-wise(K.*u)');
add_block('simulink/Math Operations/Sum',[mdl '/tauSum'],'Inputs','++++');
add_block('simulink/Discontinuities/Saturation',[mdl '/tauSat'], ...
    'UpperLimit',mat2str(sp.tauPeak,12),'LowerLimit',mat2str(-sp.tauPeak,12));
add_block('simulink/Sinks/Out1',[mdl '/tau']);  set_param([mdl '/tau'],'Port','3');
add_line(mdl,'ref/1','refDemux/1');
qSrc = 'q_mux';  dqSrc = 'dq_mux';
elastic = isfield(opts,'elastic') && ~isempty(opts.elastic);
if elastic && strcmp(saElasticParams(nAx, opts.elastic).fb, 'motor'), qSrc = 'qm_mux';  dqSrc = 'dqm_mux'; end   % 実機のエンコーダはモータ側（issue #71）
add_line(mdl,'refDemux/1','eq/1');   add_line(mdl,[qSrc '/1'],'eq/2');
add_line(mdl,'refDemux/2','edq/1');  add_line(mdl,[dqSrc '/1'],'edq/2');
add_line(mdl,'eq/1','Kp/1');         add_line(mdl,'edq/1','Kd/1');
add_line(mdl,'Kp/1','tauSum/1');     add_line(mdl,'Kd/1','tauSum/2');
add_line(mdl,'refDemux/3','tauSum/3'); add_line(mdl,'refDemux/4','tauSum/4');
add_line(mdl,'tauSum/1','tauSat/1'); add_line(mdl,'tauSat/1','tau_demux/1'); add_line(mdl,'tauSat/1','tau/1');
if elastic, set_param([mdl '/qm'],'Port','4');  set_param([mdl '/dqm'],'Port','5'); end   % 1..3 = q, dq, tau
save_system(mdl, get_param(mdl,'FileName'));
end
