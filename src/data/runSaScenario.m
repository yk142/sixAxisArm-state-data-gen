function raw = runSaScenario(s, models)
%RUNSASCENARIO 1 シナリオを 8 kHz で実行し、生データを返す。
%   raw = RUNSASCENARIO(s, models)
%     s       saScenarioSet の要素
%     models  .excite（励振モデル名）、.closed（閉ループモデル名）。ロード済みであること
%   raw: .t .q .dq .tau（8 kHz、実印加トルク）、.qref .dqref（閉ループのみ、無ければ空）、
%        .q0 .dq0（実際の初期状態）、.contact（リミット拘束 = 可動範囲の端から 0.1° 以内または外側の点の割合）
%   type ごとの内容:
%     excitation  開ループ励振＋ソフトバリア＋飽和。軸ごとに独立な信号（pattern で信号が変わる）
%     hold        PD＋FF で代表姿勢まわりの滑らかな参照に追従し、軸ごとに独立な BLN 外乱トルクを加える
%     ptp         PD＋FF による PTP 連続 GO（ランダムウェイポイント、各到達点で 0.2 s 停留）
nAx = numel(s.q0);  sp = saParams(nAx);  c = sa_config();  dt = 1/c.fsSim;
raw = struct('t',[],'q',[],'dq',[],'tau',[],'qref',[],'dqref',[],'q0',zeros(1,nAx),'dq0',zeros(1,nAx),'contact',0);
rs = RandStream('twister','Seed',s.seed);

switch s.type
    case 'excitation'
        tVec = (0:dt:s.duration)';
        amp = s.ampRel * sp.tauRated;
        u = zeros(numel(tVec), nAx);
        switch s.pattern
            case 'step'
                a = s.axis;
                u(:,a) = saSignal('step', tVec, sp.tauPeak(a), struct('tOn',0.2,'sign',1-2*(rand(rs)<0.5)));
            case 'chirp'
                for j = 1:nAx
                    u(:,j) = saSignal('chirp', tVec, amp(j), struct('f0',0.1,'f1',s.fMax,'phiDeg',360*rand(rs)));
                end
            case 'micro'      % 摩擦・粘性域: 初期姿勢の重力保持トルクのまわりに微小トルク
                u = repmat(gravityTorque(s.q0, nAx), numel(tVec), 1);
                for j = 1:nAx
                    u(:,j) = u(:,j) + saSignal('bln', tVec, amp(j), struct('fMax',s.fMax,'seed',s.seed + 101*j));
                end
            otherwise         % bln_normal / bln_high / bln_low / freefall
                for j = 1:nAx
                    u(:,j) = saSignal('bln', tVec, amp(j), struct('fMax',s.fMax,'seed',s.seed + 101*j));
                end
        end
        setSaBarrier(models.excite, sp, s.qSoftFrac, s.dqSoftFrac);
        o = simSa(models.excite, tVec, u, s.q0', s.dq0');
        raw.q0 = s.q0;  raw.dq0 = s.dq0;

    case 'hold'
        tVec = (0:dt:s.duration)';
        [qr, dqr, ddqr] = saSmoothRef(tVec, s.posture, s.refAmp, 1.0, s.seed);
        qr = min(max(qr, sp.qMin + s.refMargin), sp.qMax - s.refMargin);   % 参照を可動範囲の内側に収める
        tauD = zeros(numel(tVec), nAx);
        for j = 1:nAx
            tauD(:,j) = saSignal('bln', tVec, s.ampRel*sp.tauRated(j), struct('fMax',s.fMax,'seed',s.seed + 1 + 101*j));
        end
        REF = [tVec, qr, dqr, saFeedforward(tVec, qr, dqr, ddqr)];
        o = simSaClosed(models.closed, tVec, REF, tauD, qr(1,:)');
        raw.qref = qr;  raw.dqref = dqr;  raw.q0 = qr(1,:);

    case 'ptp'
        p = c8_params(c.meshDir);
        WP = p.qTrainMin(1:nAx) + (p.qTrainMax(1:nAx) - p.qTrainMin(1:nAx)).*rand(rs, s.nWp, nAx);
        T = saPtpSegTime(WP, s.speedScale);
        WP2 = WP(1,:);  T2 = [];                 % 各到達点で 0.2 s 停留（整定・定常偏差を評価できるように）
        for k = 1:numel(T)
            WP2(end+1,:) = WP(k+1,:);  T2(end+1,1) = T(k);      %#ok<AGROW>
            WP2(end+1,:) = WP(k+1,:);  T2(end+1,1) = 0.2;       %#ok<AGROW>
        end
        tVec = (0:dt:min(sum(T2), s.duration))';
        REF = saPtpReference(tVec, WP2, T2);
        o = simSaClosed(models.closed, tVec, REF, [], REF(1,2:1+nAx)');
        raw.qref = REF(:,2:1+nAx);  raw.dqref = REF(:,2+nAx:1+2*nAx);  raw.q0 = REF(1,2:1+nAx);

    otherwise
        error('runSaScenario:type', '未知の type: %s', s.type);
end
raw.t = o.t;  raw.q = o.q;  raw.dq = o.dq;  raw.tau = o.tau;
if strcmp(s.type,'excitation'), raw.dq0 = s.dq0; end
nearLim = raw.q <= sp.qMin + deg2rad(0.1) | raw.q >= sp.qMax - deg2rad(0.1);
raw.contact = mean(any(nearLim, 2));
end

function g = gravityTorque(q, nAx)
p = c8_params(sa_config().meshDir);  z = zeros(6,1);
t = c8RNEA([q(:); zeros(6-nAx,1)], z, z, p);  g = t(1:nAx)';
end
