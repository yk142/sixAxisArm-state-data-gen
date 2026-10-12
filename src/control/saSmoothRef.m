function [qr, dqr, ddqr] = saSmoothRef(tVec, center, amp, fMax, seed)
%SASMOOTHREF 中心まわりに滑らかにゆらぐ参照軌道（姿勢保持＋ゆっくりした設定値変動、軸ごとに独立）。
%   [qr, dqr, ddqr] = SASMOOTHREF(tVec, center, amp, fMax, seed)
%     center [1 x nAx] 中心角 [rad]、amp 振れ幅の最大値 [rad]（スカラまたは [1 x nAx]）、fMax 帯域 [Hz]、seed 乱数シード
%   200 Hz の BLN を 3 次スプライン（C2）で 8 kHz に補間し、数値微分で dq, ddq を得る（ddq_ref が不連続にならない）。
tVec = tVec(:);  nAx = numel(center);  amp = amp.*ones(1,nAx);
fsG = 200;
tG = (tVec(1):1/fsG:tVec(end)+1/fsG)';
[b,a] = butter(4, fMax/(fsG/2));
qr = zeros(numel(tVec), nAx);
for j = 1:nAx
    rs = RandStream('twister','Seed',seed + 7919*j);
    y = filtfilt(b,a,randn(rs,numel(tG),1));
    y = y / max(abs(y));
    qr(:,j) = center(j) + amp(j)*interp1(tG, y, tVec, 'spline');
end
dt = tVec(2) - tVec(1);
dqr = zeros(size(qr));  ddqr = zeros(size(qr));
for j = 1:nAx      % 行列への gradient は列方向（軸方向）の差分になるため、軸ごとに時間方向で微分する
    dqr(:,j)  = gradient(qr(:,j), dt);
    ddqr(:,j) = gradient(dqr(:,j), dt);
end
end
