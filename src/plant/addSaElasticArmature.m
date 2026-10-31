function addSaElasticArmature(mdl, p, el)
%ADDSAELASTICARMATURE 各関節に関節弾性（減速機のねじりばね＋ダンパ、モータ側慣性）を付与する（issue #71）。
%   ADDSAELASTICARMATURE(mdl, p, el)  addC8Actuation 適用済みモデルに対し、親資産の addC8Armature の代わりに呼ぶ。
%     p: c8_params（nJoints を軸数に設定済み）、el.k, el.c: 軸ごとのねじり剛性 [N*m/rad]・減衰 [N*m*s/rad]（[1 x nAx]、リンク側換算）
%   addC8Armature は反映慣性 Jm*N^2（p.armature）を RMI の 1-D ノードに直結（= 関節に剛結合）する。ここでは
%       RMI のノード R（リンク側） —[ばね k ∥ ダンパ c]— ノード M（モータ側、慣性 p.armature、理想トルク源が制御トルクを印加）
%   とし、制御トルクは関節ではなくモータ側に入る。関節（joint/t）には RMI の反作用トルクだけが入る。
%   el.stiffA・el.stiffThetaRef（issue #97、addSaUnmodeled の非線形剛性）があれば、線形ばねの代わりに saSsc.nl_spring を置く。
%   k → ∞ で addC8Armature と同じ（剛体極限）。モータ側の角度・角速度（リンク側換算）をセンサで取り、トップレベルに
%   Outport 'qm', 'dqm'（[nAx]）として出す。
n = p.nJoints;
RMI = ['fl_lib/Mechanical/Multibody Interfaces/Rotational Multibody' newline 'Interface'];
ROT = 'fl_lib/Mechanical/Rotational Elements/Inertia';
REF = ['fl_lib/Mechanical/Rotational Elements/Mechanical' newline 'Rotational Reference'];
SPR = 'fl_lib/Mechanical/Rotational Elements/Rotational Spring';
DMP = 'fl_lib/Mechanical/Rotational Elements/Rotational Damper';
TSRC = 'fl_lib/Mechanical/Mechanical Sources/Ideal Torque Source';
SENS = 'fl_lib/Mechanical/Mechanical Sensors/Ideal Rotational Motion Sensor';
PS2S = ['nesl_utility/PS-Simulink' newline 'Converter'];
add_block('simulink/Signal Routing/Mux', [mdl '/qm_mux'],  'Inputs', num2str(n), 'Position', [1000 100 1005 320]);
add_block('simulink/Signal Routing/Mux', [mdl '/dqm_mux'], 'Inputs', num2str(n), 'Position', [1000 420 1005 640]);
add_block('simulink/Sinks/Out1', [mdl '/qm'],  'Position', [1060 203 1090 217]);
add_block('simulink/Sinks/Out1', [mdl '/dqm'], 'Position', [1060 523 1090 537]);
add_line(mdl, 'qm_mux/1', 'qm/1', 'autorouting', 'on');  add_line(mdl, 'dqm_mux/1', 'dqm/1', 'autorouting', 'on');
for i = 1:n
    jb = sprintf('joint%d', i);  sc = sprintf('S2PS_%d', i);  yb = 700 + (i-1)*100;
    delete_line(mdl, [sc '/RConn1'], [jb '/LConn2']);                              % 制御トルクは関節に入れない
    rmi = sprintf('RMI_%d', i);  spr = sprintf('elSpring_%d', i);  dmp = sprintf('elDamper_%d', i);
    jm = sprintf('motorJ_%d', i);  ts = sprintf('motorTq_%d', i);  sn = sprintf('motorSns_%d', i);
    r1 = sprintf('rotRef_%d', i);  r2 = sprintf('motorRefT_%d', i);  r3 = sprintf('motorRefS_%d', i);
    add_block(RMI, [mdl '/' rmi], 'Position', [330 yb 380 yb+40]);
    if isfield(el,'stiffA') && el.stiffA ~= 0           % 非線形剛性（issue #97）: τ = k·θ·(1 + a·(θ/θr)²) の Simscape 言語の部品
        add_block('saSsc_lib/nl_spring', [mdl '/' spr], 'Position', [450 yb 490 yb+20]);
        set_param([mdl '/' spr], 'k', sprintf('%.10g', el.k(i)), 'a', sprintf('%.10g', el.stiffA), 'thr', sprintf('%.10g', el.stiffThetaRef));
    else
        add_block(SPR, [mdl '/' spr], 'Position', [450 yb 490 yb+20]);   set_param([mdl '/' spr], 'spr_rate', sprintf('%.10g', el.k(i)));
    end
    add_block(DMP, [mdl '/' dmp], 'Position', [450 yb+30 490 yb+50]); set_param([mdl '/' dmp], 'D', sprintf('%.10g', el.c(i)));
    add_block(ROT, [mdl '/' jm],  'Position', [540 yb 580 yb+30]);   set_param([mdl '/' jm], 'inertia', sprintf('%.10g', p.armature(i)));
    add_block(TSRC, [mdl '/' ts], 'Position', [540 yb+40 580 yb+70]);
    add_block(SENS, [mdl '/' sn], 'Position', [620 yb 660 yb+40]);
    add_block(REF, [mdl '/' r1], 'Position', [450 yb+85 490 yb+115]);
    add_block(REF, [mdl '/' r2], 'Position', [540 yb+85 580 yb+115]);
    add_block(REF, [mdl '/' r3], 'Position', [620 yb+85 660 yb+115]);
    % RMI: 関節の相対速度 w → RMI、反作用トルク → joint/t
    add_line(mdl, [jb '/RConn3'], [rmi '/RConn1'], 'autorouting', 'on');
    % 親資産の addC8Friction は関節トルクの供給元を tauAdd_i/RConn1 とみなす。同名の PS Add に RMI 反作用トルクと 0 を入れて互換にする
    ad = sprintf('tauAdd_%d', i);  zc = sprintf('tauZero_%d', i);
    add_block('fl_lib/Physical Signals/Functions/PS Add', [mdl '/' ad], 'Position', [230 yb 260 yb+30]);
    add_block('fl_lib/Physical Signals/Sources/PS Constant', [mdl '/' zc], 'Position', [160 yb+40 190 yb+60]);
    set_param([mdl '/' zc], 'constant', '0', 'constant_unit', 'N*m');
    add_line(mdl, [rmi '/LConn1'], [ad '/LConn1'], 'autorouting', 'on');
    add_line(mdl, [zc '/RConn1'], [ad '/LConn2'], 'autorouting', 'on');
    add_line(mdl, [ad '/RConn1'], [jb '/LConn2'], 'autorouting', 'on');
    add_line(mdl, [rmi '/RConn2'], [r1 '/LConn1'], 'autorouting', 'on');
    % リンク側ノード（RMI の R）→ ばね・ダンパ → モータ側ノード（慣性）
    add_line(mdl, [rmi '/LConn2'], [spr '/LConn1'], 'autorouting', 'on');
    add_line(mdl, [rmi '/LConn2'], [dmp '/LConn1'], 'autorouting', 'on');
    add_line(mdl, [spr '/RConn1'], [jm '/LConn1'], 'autorouting', 'on');
    add_line(mdl, [dmp '/RConn1'], [jm '/LConn1'], 'autorouting', 'on');
    % モータ側ノード: トルク源（制御トルク）とセンサ
    add_line(mdl, [sc '/RConn1'], [ts '/RConn1'], 'autorouting', 'on');
    % RMI のノード R の回転はリンク側の関節角と逆符号（実測）。モータ側ノードも同じ符号系なので、トルク源とセンサは端子を入れ替えて符号を戻す
    add_line(mdl, [ts '/LConn1'], [r2 '/LConn1'], 'autorouting', 'on');
    add_line(mdl, [ts '/RConn2'], [jm '/LConn1'], 'autorouting', 'on');
    add_line(mdl, [sn '/LConn1'], [r3 '/LConn1'], 'autorouting', 'on');
    add_line(mdl, [sn '/RConn1'], [jm '/LConn1'], 'autorouting', 'on');
    pw = sprintf('PS2S_dqm%d', i);  pq = sprintf('PS2S_qm%d', i);
    add_block(PS2S, [mdl '/' pw], 'Position', [700 yb 740 yb+14]);  add_block(PS2S, [mdl '/' pq], 'Position', [700 yb+30 740 yb+44]);
    add_line(mdl, [sn '/RConn2'], [pw '/LConn1'], 'autorouting', 'on');  add_line(mdl, [pw '/1'], sprintf('dqm_mux/%d', i), 'autorouting', 'on');
    add_line(mdl, [sn '/RConn3'], [pq '/LConn1'], 'autorouting', 'on');  add_line(mdl, [pq '/1'], sprintf('qm_mux/%d', i), 'autorouting', 'on');
end
end
