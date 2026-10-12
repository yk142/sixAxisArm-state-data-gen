function D = saDownsample(raw, fsIn, fsOut)
%SADOWNSAMPLE 8 kHz の生データを 1 kHz の遷移データへ変換する（nAx 軸）。
%   D = SADOWNSAMPLE(raw, fsIn, fsOut)   raw: .q .dq .tau [N x nAx]（8 kHz）、.qref .dqref（任意）
%
%   目的は 1 ステップ遷移 (x_k, τ_k) → x_{k+1} の教師データを作ること。
%     - 状態 (q, dq) は瞬時値をそのまま間引く（r = fsIn/fsOut 点ごと。フィルタなし）
%     - τ_k は区間 [t_k, t_{k+1}] の台形平均: 実印加トルクは 8 kHz 格子の点を直線補間（FOH）した入力なので、
%       区間の積分 / 区間長 = (τ_0/2 + τ_1 + … + τ_{r-1} + τ_r/2) / r。J2 版の単純平均（区間先頭側の r 点）より
%       終点側の寄与を正しく取る（Phase 1: 区間先頭値より 1 ステップ予測が 2.6〜8 倍良い）
%   出力（K = 遷移数）:
%     D.q, D.dq          [K+1 x nAx] 瞬時値
%     D.tau              [K x nAx] 区間の台形平均トルク、D.tauInst [K x nAx] 区間先頭の瞬時値
%     D.qref, D.dqref    [K+1 x nAx]（参照があれば。無ければ空）
%     D.fs               出力周波数
r = fsIn / fsOut;
assert(abs(r - round(r)) < 1e-9, 'saDownsample:ratio', '間引き率 %g が整数ではありません', r);
r = round(r);
N = size(raw.q, 1);  nAx = size(raw.q, 2);
K = floor((N-1) / r);
idx = 1 + r*(0:K)';
D.q  = double(raw.q(idx,:));
D.dq = double(raw.dq(idx,:));
tau = double(raw.tau);
pair = 0.5*(tau(1:end-1,:) + tau(2:end,:));           % 8 kHz 区間ごとの台形平均 [N-1 x nAx]
D.tau = reshape(permute(mean(reshape(pair(1:r*K,:), r, K, nAx), 1), [2 3 1]), K, nAx);
D.tauInst = tau(1 + r*(0:K-1)', :);
D.qref = [];  D.dqref = [];
if isfield(raw,'qref') && ~isempty(raw.qref),   D.qref  = double(raw.qref(idx,:));  end
if isfield(raw,'dqref') && ~isempty(raw.dqref), D.dqref = double(raw.dqref(idx,:)); end
D.fs = fsOut;
end
