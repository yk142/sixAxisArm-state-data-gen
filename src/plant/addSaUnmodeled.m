function um = addSaUnmodeled(mdl, p, el, um)
%ADDSAUNMODELED 解析モデル（2 慣性解析）の式にない効果を真値に加える（issue #97）。
%   el = saUnmodeledElastic(el, um) で非線形剛性を el に載せてから addSaElasticArmature を呼び（ばねの部品を置き換える）、
%   addC8Friction の後に ADDSAUNMODELED(mdl, p, el, um) で Stribeck 摩擦を足す。
%     p: c8_params（nJoints 設定済み）、el: saElasticParams の出力、um: 構造体（省略した項目は既定値）
%   1) 非線形剛性（硬化ばね、saUnmodeledElastic と addSaElasticArmature）: 線形ばね k·θ を τ = k·θ·(1 + a·(θ/θr)²) の部品に置き換える（θ = リンク側 − モータ側のねじれ角）。
%      合計 τs = k·θ·(1 + a·(θ/θr)²)。um.stiffA = a（既定 0.3）、um.stiffThetaRef = θr [rad]（既定 0.005）。a = 0 で無効
%      減速機（波動歯車など）はねじれが大きいほど硬い。θ = θr で割線剛性が (1 + a) 倍
%   2) Stribeck 摩擦: 関節（リンク側）の摩擦に τst = Fs·exp(−(ω/vs)²)·tanh(ω/ε) を足す（運動を妨げる向き）。
%      Fs = um.stribeck·Fc（既定 0.3）、vs = um.vsMul·ε（既定 2）。低速で摩擦が増える（静止摩擦 > 動摩擦）。0 で無効
%   非線形剛性は Simscape 言語の部品（src/plant/ssc/+saSsc/nl_spring.ssc。初回に ssc_build で saSsc_lib.slx を作る）。
%   （センサとトルク源で組むと Simulink の代数ループになり符号も端子の向きに依存した。追加項だけの部品を並列に置いても状態が重複して代数ループになったので、ばね全体を置き換える）
%   Stribeck のトルクは addSaElasticArmature が作る tauAdd_i の 2 番目の入力（0 の定数）を置き換えて関節に入れる。
if nargin < 4, um = struct(); end
um = saUnmodeledDefaults(um);
n = p.nJoints;
GAIN  = 'fl_lib/Physical Signals/Functions/PS Gain';
MATHF = 'fl_lib/Physical Signals/Functions/PS Math Function';
PROD  = 'fl_lib/Physical Signals/Functions/PS Product';
for i = 1:n
    yb = 2200 + (i-1)*160;
    if um.stribeck ~= 0
        % --- Stribeck: ω → (ω/vs)² → ×(−1) → exp ； ω/ε → tanh ； 積 × (−Fs) → tauAdd_i の 2 番目の入力 ---
        jb = sprintf('joint%d', i);  epsI = p.frictionEps(i);  vs = um.vsMul*epsI;  Fs = um.stribeck*p.frictionFc(i);
        a1 = sprintf('stW_%d', i);  a2 = sprintf('stSq_%d', i);  a3 = sprintf('stNeg_%d', i);  a4 = sprintf('stExp_%d', i);
        b1 = sprintf('stWe_%d', i);  b2 = sprintf('stTanh_%d', i);  pr = sprintf('stProd_%d', i);  fg = sprintf('stFs_%d', i);
        add_block(GAIN, [mdl '/' a1], 'Position', [330 yb+60 370 yb+90]);  set_param([mdl '/' a1], 'gain', sprintf('%.10g', 1/vs), 'gain_unit', 's/rad');
        add_block(MATHF, [mdl '/' a2], 'Position', [400 yb+60 440 yb+90]);  set_param([mdl '/' a2], 'fcn_choice', 'foundation.enum.math_function.square');
        add_block(GAIN, [mdl '/' a3], 'Position', [470 yb+60 510 yb+90]);  set_param([mdl '/' a3], 'gain', '-1', 'gain_unit', '1');
        add_block(MATHF, [mdl '/' a4], 'Position', [540 yb+60 580 yb+90]);  set_param([mdl '/' a4], 'fcn_choice', 'foundation.enum.math_function.exp');
        add_block(GAIN, [mdl '/' b1], 'Position', [330 yb+110 370 yb+140]); set_param([mdl '/' b1], 'gain', sprintf('%.10g', 1/epsI), 'gain_unit', 's/rad');
        add_block(MATHF, [mdl '/' b2], 'Position', [400 yb+110 440 yb+140]); set_param([mdl '/' b2], 'fcn_choice', 'foundation.enum.math_function.tanh');
        add_block(PROD, [mdl '/' pr], 'Position', [610 yb+80 640 yb+110]);
        add_block(GAIN, [mdl '/' fg], 'Position', [670 yb+80 710 yb+110]);  set_param([mdl '/' fg], 'gain', sprintf('%.10g', -Fs), 'gain_unit', 'N*m');
        add_line(mdl, [jb '/RConn3'], [a1 '/LConn1'], 'autorouting', 'on');
        add_line(mdl, [a1 '/RConn1'], [a2 '/LConn1'], 'autorouting', 'on');
        add_line(mdl, [a2 '/RConn1'], [a3 '/LConn1'], 'autorouting', 'on');
        add_line(mdl, [a3 '/RConn1'], [a4 '/LConn1'], 'autorouting', 'on');
        add_line(mdl, [jb '/RConn3'], [b1 '/LConn1'], 'autorouting', 'on');
        add_line(mdl, [b1 '/RConn1'], [b2 '/LConn1'], 'autorouting', 'on');
        add_line(mdl, [a4 '/RConn1'], [pr '/LConn1'], 'autorouting', 'on');
        add_line(mdl, [b2 '/RConn1'], [pr '/LConn2'], 'autorouting', 'on');
        add_line(mdl, [pr '/RConn1'], [fg '/LConn1'], 'autorouting', 'on');
        ad = sprintf('tauAdd_%d', i);  zc = sprintf('tauZero_%d', i);
        delete_line(mdl, [zc '/RConn1'], [ad '/LConn2']);
        delete_block([mdl '/' zc]);
        add_line(mdl, [fg '/RConn1'], [ad '/LConn2'], 'autorouting', 'on');
    end
end
end
