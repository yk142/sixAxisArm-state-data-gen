function [REF, tauD] = saClosedInputs(s)
%SACLOSEDINPUTS 閉ループシナリオ（hold / ptp）の入力を作り直す（runSaScenario と同じ式。シミュレーションはしない）。
%   [REF, tauD] = SACLOSEDINPUTS(s)   s: saScenarioSet の要素（type が 'hold' または 'ptp'）
%   REF  [N x (1+3·nAx)] = [t, q_ref, dq_ref, tau_ff]（8 kHz 格子）、tauD [N x nAx] 外乱トルク（ptp は 0）
%   閉ループ（buildSaClosedLoop）の入力そのもので、Python 側の閉ループ評価（制御器の移植）が使う（issue #45）。
nAx = numel(s.q0);  sp = saParams(nAx);  c = sa_config();  dt = 1/c.fsSim;
rs = RandStream('twister','Seed',s.seed);
switch s.type
    case 'hold'
        tVec = (0:dt:s.duration)';
        [qr, dqr, ddqr] = saSmoothRef(tVec, s.posture, s.refAmp, 1.0, s.seed);
        if isfield(s,'pushDepth') && s.pushDepth > 0
            % リミット押し付け（issue #47）: 目標軸の参照を、1.0〜2.5 s の間に滑らかに「リミットの pushDepth 外側」を中心とする振れへ移す
            a = s.axis;  side = 2*(sp.qMax(a) - s.posture(a) < s.posture(a) - sp.qMin(a)) - 1;     % 姿勢が近い側のリミット（+1: 上限）
            lim = ternary(side > 0, sp.qMax(a), sp.qMin(a));
            w = smoothstep((tVec - 1.0)/1.5);
            qr(:,a) = qr(:,a) + w.*(lim + side*s.pushDepth - s.posture(a));
            o = setdiff(1:nAx, a);  qr(:,o) = min(max(qr(:,o), sp.qMin(o) + s.refMargin), sp.qMax(o) - s.refMargin);
            for j = a
                dqr(:,j) = gradient(qr(:,j), dt);  ddqr(:,j) = gradient(dqr(:,j), dt);
            end
        else
            qr = min(max(qr, sp.qMin + s.refMargin), sp.qMax - s.refMargin);
        end
        tauD = zeros(numel(tVec), nAx);
        for j = 1:nAx
            tauD(:,j) = saSignal('bln', tVec, s.ampRel*sp.tauRated(j), struct('fMax',s.fMax,'seed',s.seed + 1 + 101*j));
        end
        REF = [tVec, qr, dqr, saFeedforward(tVec, qr, dqr, ddqr)];
    case 'ptp'
        p = c8_params(c.meshDir);
        WP = p.qTrainMin(1:nAx) + (p.qTrainMax(1:nAx) - p.qTrainMin(1:nAx)).*rand(rs, s.nWp, nAx);
        T = saPtpSegTime(WP, s.speedScale);
        WP2 = WP(1,:);  T2 = [];
        for k = 1:numel(T)
            WP2(end+1,:) = WP(k+1,:);  T2(end+1,1) = T(k);      %#ok<AGROW>
            WP2(end+1,:) = WP(k+1,:);  T2(end+1,1) = 0.2;       %#ok<AGROW>
        end
        tVec = (0:dt:min(sum(T2), s.duration))';
        REF = saPtpReference(tVec, WP2, T2);
        tauD = zeros(numel(tVec), nAx);
    otherwise
        error('saClosedInputs:type', 'hold または ptp のみ: %s', s.type);
end
end

function y = ternary(c, a, b)
if c, y = a; else, y = b; end
end

function y = smoothstep(x)
x = min(max(x, 0), 1);  y = x.^2.*(3 - 2*x);
end
