function tauFf = saFeedforward(tVec, qr, dqr, ddqr, payload)
%SAFEEDFORWARD 参照軌道（q, dq, ddq）から RNEA＋摩擦のフィードフォワードトルクを計算する（nAx 軸）。
%   tauFf = SAFEEDFORWARD(tVec, qr, dqr, ddqr [, payload])   各 [n x nAx]。payload（struct: mass, offset）を渡すと手先ペイロード込み。固定軸（nAx+1..6）は 0 rad。
%   RNEA は重い（1 呼び出し約 6 ms）ため約 400 Hz で評価して線形補間する（親資産 c8FeedForward と同じ方針。
%   参照が滑らかなので補間差は小さく、PD が吸収する）。
tVec = tVec(:);  n = numel(tVec);  nAx = size(qr,2);
p = c8_params(sa_config().meshDir);
if nargin >= 5 && ~isempty(payload), p = saApplyPayload(p, payload); end
Bv = p.frictionBv(1:nAx);  Fc = p.frictionFc(1:nAx);  ep = p.frictionEps(1:nAx);
dtIn = median(diff(tVec));
coarse = max(1, round((1/400)/dtIn));
ic = unique([1:coarse:n, n]);
Tc = zeros(numel(ic), nAx);
for m = 1:numel(ic)
    i = ic(m);
    q6 = [qr(i,:)'; zeros(6-nAx,1)];  dq6 = [dqr(i,:)'; zeros(6-nAx,1)];  ddq6 = [ddqr(i,:)'; zeros(6-nAx,1)];
    t6 = c8RNEA(q6, dq6, ddq6, p)';
    Tc(m,:) = t6(1:nAx) + Bv.*dqr(i,:) + Fc.*tanh(dqr(i,:)./ep);
end
tauFf = interp1(tVec(ic), Tc, tVec, 'linear', 'extrap');
end
