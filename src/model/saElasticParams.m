function el = saElasticParams(nAx, e)
%SAELASTICPARAMS 関節弾性のパラメータ（ねじり剛性 k、減衰 c、フィードバック位置）を決める（issue #71）。実機値は未入手で、仮の代表値。
%   el = SAELASTICPARAMS(nAx, e)   e: 構造体。次のいずれか
%     e.f [Hz] , e.zeta : 軸ごと（スカラ可）に「リンク側慣性 Jl とモータ側慣性 Jm の 2 慣性系の固有振動数」を f にする剛性を決める。
%                         Jl = max(Mmean − armature, 0.2·armature)（Mmean は 0 kg の平均）。減衰比は zeta（既定 0.05）
%     e.k , e.c         : 直接指定 [1 x nAx]（k を指定すると f は無視。c 省略時は zeta から決める）
%     e.thetaMax [rad]  : 定格トルクでのねじれ角の上限（既定 0.02 rad ≒ 1.1°）。f から決めた k がこれより柔らかい軸（慣性が小さい手首側）は k = tauRated/thetaMax に引き上げる
%     e.kScale          : k に掛ける倍率（感度用、既定 1）。c も同じ減衰比になるよう再計算
%     e.fb              : 'motor'（既定。PD のフィードバックをモータ側角度・速度にする = エンコーダ位置）または 'link'
%   出力 el.k, el.c, el.fb, el.Jl, el.Jm, el.f（リンク慣性 Jl での 2 慣性系の固有振動数 [Hz]）
sp = saParams(nAx);
Jm = sp.armature;  Jl = max(sp.Mmean - Jm, 0.2*Jm);  Jr = Jl.*Jm./(Jl + Jm);
zeta = 0.05;  if isfield(e,'zeta') && ~isempty(e.zeta), zeta = e.zeta; end
if isfield(e,'k') && ~isempty(e.k), k = e.k.*ones(1,nAx);
else
    f = 20;  if isfield(e,'f') && ~isempty(e.f), f = e.f; end
    k = (2*pi*f.*ones(1,nAx)).^2.*Jr;
end
thetaMax = 0.02;  if isfield(e,'thetaMax') && ~isempty(e.thetaMax), thetaMax = e.thetaMax; end
if ~(isfield(e,'k') && ~isempty(e.k)), k = max(k, sp.tauRated/thetaMax); end
if isfield(e,'kScale') && ~isempty(e.kScale), k = k.*e.kScale; end
if isfield(e,'c') && ~isempty(e.c), c = e.c.*ones(1,nAx); else, c = 2*zeta.*sqrt(k.*Jr); end
el.k = k;  el.c = c;  el.Jl = Jl;  el.Jm = Jm;  el.f = sqrt(k./Jr)/(2*pi);
el.fb = 'motor';  if isfield(e,'fb') && ~isempty(e.fb), el.fb = e.fb; end
end
