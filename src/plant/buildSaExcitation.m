function mdl = buildSaExcitation(nAx, opts)
%BUILDSAEXCITATION 開ループ励振モデル（励振トルク＋軸ごとのソフトバリア＋トルク飽和）を生成する。
%   mdl = BUILDSAEXCITATION(nAx)         nAx = 3 または 6。data/models/sa_excite<nAx>.slx に保存
%   mdl = BUILDSAEXCITATION(nAx, opts)   opts は buildSaPlant のものに加え:
%       opts.barrierWn  バリアの固有角周波数 [rad/s]（既定 60）
%   I/F:  Inport 'tauEx' [nAx] → Outport 'q', 'dq', 'tau'（飽和後の実印加トルク）、opts.elastic 指定時は 4, 5 = 'qm', 'dqm'（issue #73）
%
%   ソフトバリア（J2 版・親資産 buildC8Excitation と同じ考え方、軸ごと）:
%       τ = sat( τ_ex − Kp_bar·deadzone(q; qSoftMin, qSoftMax) − Kd_bar·deadzone(dq; ±dqSoft) )
%   Dead Zone は安全帯の内側で厳密に 0 を返すため、帯の内側では励振の分布を歪めない。
%   ゲインは軸ごとの質量行列の対角の平均 Mmean から決める:
%       Kp = max(M·wn², 1.5·tauPeak/margin)、Kd = max(2·M·wn, tauPeak/(0.2·qdMax))。
%   Kp の第 2 項（margin = 既定の安全帯（位置 0.85）の端からリミットまでの角度）は、リミット位置でのバリアトルクが
%   tauPeak の 1.5 倍以上になる値。慣性の小さい手首 J5 などで、tauPeak の励振がリミットに到達しないようにする
%   （J1〜J3 では第 1 項が大きく、値は変わらない）。
%   Kd の第 2 項は、速度が安全帯の上限を最大速度の 20% 超えたところでトルク飽和に達する値で、
%   慣性が小さく tauPeak が大きい軸（J3 など）でも速度が qdMax を大きく超えないようにする。
%   飽和は軸ごとの瞬時最大トルク tauPeak。安全帯は setSaBarrier で実行前に変更できる（既定: 位置 0.85、速度 0.60）。
if nargin < 2, opts = struct(); end
if ~isfield(opts,'modelName') || isempty(opts.modelName), opts.modelName = sprintf('sa_excite%d',nAx); end
if ~isfield(opts,'barrierWn') || isempty(opts.barrierWn), opts.barrierWn = 60; end
bw = opts.barrierWn;  opts = rmfield(opts,'barrierWn');
mdl = buildSaPlant(nAx, opts);
sp = saParams(nAx);

set_param([mdl '/tau'],'Name','tauEx');
delete_line(mdl,'tauEx/1','tau_demux/1');
add_block('simulink/Discontinuities/Dead Zone',[mdl '/qDead'], 'LowerValue','-1','UpperValue','1');
add_block('simulink/Discontinuities/Dead Zone',[mdl '/dqDead'],'LowerValue','-1','UpperValue','1');
add_block('simulink/Math Operations/Gain',[mdl '/KpBar'],'Gain',mat2str(-max(sp.Mmean*bw^2, 1.5*sp.tauPeak./((1-0.85)*(sp.qMax-sp.qMin)/2)),12),'Multiplication','Element-wise(K.*u)');
add_block('simulink/Math Operations/Gain',[mdl '/KdBar'],'Gain',mat2str(-max(2*sp.Mmean*bw, sp.tauPeak./(0.2*sp.qdMax)),12),'Multiplication','Element-wise(K.*u)');
add_block('simulink/Math Operations/Sum',[mdl '/tauSum'],'Inputs','+++');
add_block('simulink/Discontinuities/Saturation',[mdl '/tauSat'], ...
    'UpperLimit',mat2str(sp.tauPeak,12),'LowerLimit',mat2str(-sp.tauPeak,12));
elastic = isfield(opts,'elastic') && ~isempty(opts.elastic);
if elastic, set_param([mdl '/qm'],'Port','4');  set_param([mdl '/dqm'],'Port','5'); end   % 関節弾性つき（issue #73）: 1..3 = q, dq, tau
add_block('simulink/Sinks/Out1',[mdl '/tau']);  set_param([mdl '/tau'],'Port','3');
add_line(mdl,'q_mux/1','qDead/1');   add_line(mdl,'qDead/1','KpBar/1');
add_line(mdl,'dq_mux/1','dqDead/1'); add_line(mdl,'dqDead/1','KdBar/1');
add_line(mdl,'tauEx/1','tauSum/1');  add_line(mdl,'KpBar/1','tauSum/2'); add_line(mdl,'KdBar/1','tauSum/3');
add_line(mdl,'tauSum/1','tauSat/1'); add_line(mdl,'tauSat/1','tau_demux/1'); add_line(mdl,'tauSat/1','tau/1');
setSaBarrier(mdl, sp, 0.85, 0.60);
save_system(mdl, get_param(mdl,'FileName'));
end
