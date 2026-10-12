function REF = saPtpReference(tVec, WP, T)
%SAPTPREFERENCE PTP（5 次多項式・マルチウェイポイント）の参照軌道とフィードフォワードトルクを事前計算する。
%   REF = SAPTPREFERENCE(tVec, WP, T)
%     tVec  時刻ベクトル [n x 1]（8 kHz 格子）、WP [(nSeg+1) x nAx] [rad]、T [nSeg x 1] [s]
%     REF   [n x (1 + 3·nAx)] = [t, q_ref, dq_ref, tau_ff]
%   親資産 c8FeedForward（RNEA＋摩擦、約 400 Hz で評価して線形補間）を、固定軸を 0 rad とした 6 軸で
%   呼び、先頭 nAx 軸を取り出す。固定軸のトルクは印加されないため、可動軸の τ_ff は nAx 軸の RNEA と一致する。
%   姿勢保持は、同じ姿勢を 2 点並べた WP と T = 0.1 で呼ぶ（親の c8TrajGen はセグメントが 1 つ以上要る）。
%   このとき q_ref 一定、dq_ref = 0、τ_ff = 重力補償になる。WP が 1 点なら自動でそうする。
nAx = size(WP,2);  p = c8_params(sa_config().meshDir);
if size(WP,1) == 1, WP = [WP; WP];  T = 0.1; end
WP6 = [WP, zeros(size(WP,1), 6-nAx)];
R = c8FeedForward(tVec(:), WP6, T(:), p);
REF = [R(:,1), R(:,2:1+nAx), R(:,8:7+nAx), R(:,14:13+nAx)];
end
