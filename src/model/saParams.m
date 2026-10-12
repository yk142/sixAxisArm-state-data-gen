function sp = saParams(nAx)
%SAPARAMS 先頭 nAx 軸（3 または 6）の導出パラメータを返す（バリア・励振振幅・ゲインの基準に使う）。
%   sp = SAPARAMS(nAx)
%   出力フィールド（単位は SI、軸ごとの [1 x nAx]）:
%     qMin, qMax, qdMax      可動範囲 [rad]、最大角速度 [rad/s]
%     tauRated, tauPeak      関節側の定格・瞬時最大トルク [N*m]
%     Mmean, Mmin, Mmax      質量行列の対角（armature 込み）の、[0.6 qMin, 0.6 qMax] 内の乱数姿勢 200 点での平均・最小・最大
%                            [kg*m^2]。固定軸（nAx+1..6）は 0 rad
%     armature, Fc, Bv, eps  armature [kg*m^2]、摩擦係数、tanh 幅
%   質量行列は姿勢で大きく変わる（特に J1〜J3）ため、バリアのゲインは平均値 Mmean から決める。
c = sa_config();  p = c8_params(c.meshDir);
idx = 1:nAx;
sp.nAx = nAx;
sp.qMin = p.qMin(idx);  sp.qMax = p.qMax(idx);  sp.qdMax = p.qdMax(idx);
sp.tauRated = p.tauRated(idx);  sp.tauPeak = p.tauPeak(idx);
sp.armature = p.armature(idx);
sp.Fc = p.frictionFc(idx);  sp.Bv = p.frictionBv(idx);  sp.eps = p.frictionEps(idx);
rs = RandStream('mt19937ar','Seed',1);
D = zeros(200, nAx);  z = zeros(6,1);
for k = 1:200
    q = [0.6*p.qMin(idx)' + 0.6*(p.qMax(idx)-p.qMin(idx))'.*rand(rs,nAx,1); zeros(6-nAx,1)];   % [0.6 qMin, 0.6 qMax] の一様乱数
    g = c8RNEA(q, z, z, p);
    for j = idx
        e = z;  e(j) = 1;
        m = c8RNEA(q, z, e, p) - g;
        D(k,j) = m(j);
    end
end
sp.Mmean = mean(D,1);  sp.Mmin = min(D,[],1);  sp.Mmax = max(D,[],1);
end
