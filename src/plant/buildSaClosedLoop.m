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
%   opts.unmodeled（issue #97）があるときは、PD のフィードバック（−Kp·qm − Kd·dqm）と飽和を Simscape の物理信号（PS）の中で組む
%   （pdToPhysical）。非線形ばねの部品を入れると Simulink と Simscape の間のフィードバックが代数ループになり計算が約 4.5 倍遅くなったため。
%   式は同じ。この場合 Kp・Kd は構築時の値で固定（set_param の Kp・Kd は参照項 Kp·q_ref + Kd·dq_ref にしか効かない）。
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
if isfield(opts,'unmodeled') && ~isempty(opts.unmodeled)
    pdToPhysical(mdl, nAx, sp.Mmean.*wn.^2, 2*zeta*sp.Mmean.*wn, sp.tauPeak);
end
save_system(mdl, get_param(mdl,'FileName'));
end


function pdToPhysical(mdl, nAx, Kp, Kd, peak)
% PD のフィードバックと飽和を物理信号の中に移す（issue #97）。Simulink 側は u = tau_ff + tau_dist + Kp·q_ref + Kd·dq_ref だけを計算し、
% Simscape 側で τ = sat(u − Kp·qm − Kd·dqm)（qm, dqm はモータ側センサ motorSns_i の角度・角速度）をトルク源 motorTq_i に入れる。
GAIN = 'fl_lib/Physical Signals/Functions/PS Gain';  PSADD = 'fl_lib/Physical Signals/Functions/PS Add';
SAT = 'fl_lib/Physical Signals/Nonlinear Operators/PS Saturation';  PS2S = ['nesl_utility/PS-Simulink' newline 'Converter'];
% Simulink 側: 偏差（eq, edq）を参照そのものに置き換え、飽和を外す
delete_line(mdl,'eq/1','Kp/1');  delete_line(mdl,'edq/1','Kd/1');
add_line(mdl,'refDemux/1','Kp/1');  add_line(mdl,'refDemux/2','Kd/1');
delete_line(mdl,'tauSat/1','tau_demux/1');  delete_line(mdl,'tauSat/1','tau/1');  delete_line(mdl,'tauSum/1','tauSat/1');
delete_block([mdl '/tauSat']);
add_line(mdl,'tauSum/1','tau_demux/1');
for b = {'eq','edq'}
    lh = get_param([mdl '/' b{1}],'LineHandles');  delete_line(lh.Inport(ishandle(lh.Inport)));  delete_block([mdl '/' b{1}]);
end
add_block('simulink/Signal Routing/Mux', [mdl '/tauOut_mux'], 'Inputs', num2str(nAx), 'Position', [1000 760 1005 980]);
add_line(mdl, 'tauOut_mux/1', 'tau/1', 'autorouting', 'on');
for i = 1:nAx
    yb = 3300 + (i-1)*120;  sc = sprintf('S2PS_%d', i);  ts = sprintf('motorTq_%d', i);  sn = sprintf('motorSns_%d', i);
    gp = sprintf('pdKp_%d', i);  gd = sprintf('pdKd_%d', i);  a1 = sprintf('pdAdd1_%d', i);  a2 = sprintf('pdAdd2_%d', i);  st = sprintf('pdSat_%d', i);  po = sprintf('PS2S_tau%d', i);
    delete_line(mdl, [sc '/RConn1'], [ts '/RConn1']);
    add_block(GAIN, [mdl '/' gp], 'Position', [330 yb 370 yb+30]);     set_param([mdl '/' gp], 'gain', sprintf('%.12g', -Kp(i)), 'gain_unit', 'N*m/rad');
    add_block(GAIN, [mdl '/' gd], 'Position', [330 yb+40 370 yb+70]);  set_param([mdl '/' gd], 'gain', sprintf('%.12g', -Kd(i)), 'gain_unit', 'N*m*s/rad');
    add_block(PSADD, [mdl '/' a1], 'Position', [400 yb 430 yb+30]);
    add_block(PSADD, [mdl '/' a2], 'Position', [460 yb 490 yb+30]);
    add_block(SAT, [mdl '/' st], 'Position', [520 yb 560 yb+30]);
    set_param([mdl '/' st], 'upper_limit', sprintf('%.12g', peak(i)), 'upper_limit_unit', 'N*m', 'lower_limit', sprintf('%.12g', -peak(i)), 'lower_limit_unit', 'N*m');
    add_block(PS2S, [mdl '/' po], 'Position', [600 yb 640 yb+14]);
    add_line(mdl, [sn '/RConn3'], [gp '/LConn1'], 'autorouting', 'on');     % モータ側角度
    add_line(mdl, [sn '/RConn2'], [gd '/LConn1'], 'autorouting', 'on');     % モータ側角速度
    add_line(mdl, [gp '/RConn1'], [a1 '/LConn1'], 'autorouting', 'on');
    add_line(mdl, [gd '/RConn1'], [a1 '/LConn2'], 'autorouting', 'on');
    add_line(mdl, [sc '/RConn1'], [a2 '/LConn1'], 'autorouting', 'on');     % u（Simulink から）
    add_line(mdl, [a1 '/RConn1'], [a2 '/LConn2'], 'autorouting', 'on');
    add_line(mdl, [a2 '/RConn1'], [st '/LConn1'], 'autorouting', 'on');
    add_line(mdl, [st '/RConn1'], [ts '/RConn1'], 'autorouting', 'on');
    add_line(mdl, [st '/RConn1'], [po '/LConn1'], 'autorouting', 'on');
    add_line(mdl, [po '/1'], sprintf('tauOut_mux/%d', i), 'autorouting', 'on');
end
end
