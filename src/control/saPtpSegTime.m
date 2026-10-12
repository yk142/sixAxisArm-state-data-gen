function T = saPtpSegTime(WP, speedScale)
%SAPTPSEGTIME 速度制限からPTPの各セグメント所要時間を決める（親資産 c8TrajSegTime の nAx 軸版）。
%   T = SAPTPSEGTIME(WP, speedScale)   WP [(nSeg+1) x nAx] [rad]。5 次多項式のピーク速度 1.875·Δq/T が
%   speedScale·qdMax 以下になる最小の T（下限 0.05 s）。
nAx = size(WP,2);  p = c8_params(sa_config().meshDir);
T = c8TrajSegTime([WP, zeros(size(WP,1), 6-nAx)], p, speedScale);
end
