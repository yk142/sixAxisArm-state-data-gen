function S = saScenarioSet(nAx, preset)
%SASCENARIOSET 学習データ生成のシナリオ表を返す（docs/phase2_design.md。J2 版の構成を軸ごとに拡張）。
%   S = SASCENARIOSET(nAx, preset)   nAx = 3 | 6、preset = 'full'（既定）| 'small'（動作確認用）
%
%   S は構造体配列。主なフィールド:
%     name, pattern, type ('excitation'|'hold'|'ptp'), phase (1|2|'benchmark'), split, seed, duration [s]
%     ampRel   励振振幅 / 軸ごとの定格トルク、fMax 帯域 [Hz]（軸ごとに独立な信号。軸間相関は 0）
%     q0, dq0  初期状態 [1 x nAx]（励振。hold / ptp は参照軌道から決まる）
%     qSoftFrac, dqSoftFrac  ソフトバリアの安全帯比率（励振のみ。リミット接触は Phase 3 の追加枠）
%     posture  hold の中心姿勢 [1 x nAx]、refAmp 参照の振れ幅 [rad]、refMargin 可動範囲端からの余裕 [rad]
%     axis     step を与える軸（step のみ）、speedScale / nWp  ptp のパラメータ
%   full（3 軸）: 158 本（＋リミット接触 60 本）、励振・保持・PTP の継続時間の合計 約 1,290 s（J2 版の約半分。設計書の規模）。
%   初期状態は Sobol 列（2·nAx 次元、digital shift で seed 固定）で、バリア帯の内側・リミットから 2° 以上内側に取る。
%   train / val / test はパターンごとに 70 / 15 / 15（シード固定）、PTP は 'benchmark'。
assert(any(nAx == [3 6]), 'saScenarioSet:nAx', 'nAx は 3 または 6');
if nargin < 2, preset = 'full'; end
c = sa_config();  sp = saParams(nAx);  p = c8_params(c.meshDir);
d2r = pi/180;

qLo = sp.qMin + (1-0.85)*(sp.qMax-sp.qMin)/2 + 2*d2r;     % 初期状態の範囲: バリア帯（0.85）の内側、リミットから 2° 以上
qHi = sp.qMax - (1-0.85)*(sp.qMax-sp.qMin)/2 - 2*d2r;
dqI = 0.6 * sp.qdMax * 0.8;                               % 初期角速度の範囲（速度バリア帯の 80%）

% 代表姿勢（J1〜J6 [rad]。J2 は重力の効きが大きく変わる範囲、J3 は肘の曲げ。叩き台）
%   手首 J4〜J6: p1〜p4 は J5 を 0 から 0.5 rad 以上離した通常姿勢、p5 は特異点近傍（J5 = 0.2 rad）、p6 は J5 = 0（特異点）。
%   3 軸では先頭 3 列のみ使う。
postures = [ 0.0 -0.5  0.5   0.4  0.8 -0.4
             0.8 -1.2  1.5  -0.5 -0.9  0.6
            -0.8  0.5  0.3   0.3  1.2 -0.3
             1.5 -1.9  2.0  -0.4 -0.5  0.5
            -1.5 -0.3  1.0   0.0  0.2  0.0
             0.0 -2.2  1.0   0.0  0.0  0.0];
postures = postures(:, 1:nAx);

% ---- パターン定義: {pattern, type, phase, 件数, 継続時間, ampRel, fMax} ----
switch preset
    case 'full'
        P = { 'bln_normal','excitation',1, 24,  8, 0.5, 20
              'bln_high',  'excitation',1, 12,  4, 1.0, 20
              'bln_low',   'excitation',1, 12, 10, 0.2,  5
              'freefall',  'excitation',1,  6,  3, 0.05, 5
              'chirp',     'excitation',2,  8, 15, 0.5, 20
              'step',      'excitation',2, 12,  2, 1.0, 0
              'micro',     'excitation',2, 12,  8, 0.05, 5 };
        nHold = 8;  durHold = 5;  nNear = 2;  durNear = 6;  nPtp = 3;  speeds = [0.2 0.4 0.6 0.8];  nWp = 5;  durPtp = 30;
        iPost = 1:size(postures,1);
    case 'small'
        P = { 'bln_normal','excitation',1, 2, 2, 0.5, 20
              'bln_high',  'excitation',1, 1, 1, 1.0, 20
              'bln_low',   'excitation',1, 1, 2, 0.2,  5
              'freefall',  'excitation',1, 1, 1.5, 0.05, 5
              'chirp',     'excitation',2, 1, 3, 0.5, 20
              'step',      'excitation',2, nAx, 1, 1.0, 0
              'micro',     'excitation',2, 1, 2, 0.05, 5 };
        nHold = 1;  durHold = 2;  nNear = 1;  durNear = 2;  nPtp = 1;  speeds = [0.3 0.7];  nWp = 3;  durPtp = 10;
        iPost = [1 2];  if nAx == 6, iPost = [1 2 5 6]; end      % 6 軸は特異点近傍（p5）・特異点（p6）も含める
    otherwise
        error('saScenarioSet:preset', '未知の preset: %s', preset);
end

nExc = sum([P{:,4}]);
U = saSobol(nExc, 2*nAx, c.seedBase);
S = struct([]);
k = 0;
for r = 1:size(P,1)
    for i = 1:P{r,4}
        k = k + 1;
        s = emptyScenario(nAx);
        s.pattern = P{r,1};  s.type = P{r,2};  s.phase = P{r,3};
        s.name = sprintf('%s_%03d', s.pattern, i);
        s.seed = c.seedBase + 1000*r + i;
        s.duration = P{r,5};  s.ampRel = P{r,6};  s.fMax = P{r,7};
        s.q0  = qLo + U(k,1:nAx).*(qHi - qLo);
        s.dq0 = (2*U(k,nAx+1:end) - 1) .* dqI;
        if strcmp(s.pattern,'freefall'), s.dq0 = s.dq0 * 0.5; end
        if strcmp(s.pattern,'step'), s.axis = mod(i-1, nAx) + 1; end      % step を与える軸を巡回
        S = [S, s]; %#ok<AGROW>
    end
end

% ---- 姿勢保持 + 外乱（閉ループ PD+FF、代表姿勢）----
for ip = iPost
    for i = 1:nHold
        s = emptyScenario(nAx);
        s.pattern = sprintf('hold_p%d', ip);  s.type = 'hold';  s.phase = 2;
        s.name = sprintf('%s_%03d', s.pattern, i);
        s.seed = c.seedBase + 100000 + 1000*ip + i;
        s.duration = durHold;  s.ampRel = 0.3;  s.fMax = 20;
        s.posture = postures(ip,:);  s.refAmp = deg2rad(10);  s.refMargin = deg2rad(3);
        S = [S, s]; %#ok<AGROW>
    end
end

% ---- リミット近傍（閉ループ保持。軸ごと・両側。参照はリミットの 4° 内側、振れ幅 ±3°、強めの外乱）----
base = [0 -0.5 0.8 0 0.6 0];  base = base(1:nAx);          % 他軸の姿勢（J5 は特異点から離す）
k = 0;
for i = 1:nNear*nAx*2
    s = emptyScenario(nAx);
    ax = mod(i-1, nAx) + 1;  side = 1 - 2*(mod(floor((i-1)/nAx), 2) == 1);   % 軸を巡回し、次の周回で反対側
    s.pattern = 'nearlimit';                                 % split は軸・側をまとめて 1 パターンとして行う
    s.type = 'hold';  s.phase = 2;
    s.name = sprintf('nearlimit_j%d%s_%03d', ax, ternary(side > 0, 'p', 'm'), ceil(i/(nAx*2)));
    s.seed = c.seedBase + 150000 + i;
    post = base;
    if side > 0, post(ax) = sp.qMax(ax) - 4*d2r; else, post(ax) = sp.qMin(ax) + 4*d2r; end
    s.posture = post;  s.duration = durNear;  s.ampRel = 0.5;  s.fMax = 20;
    s.refAmp = 3*d2r;  s.refMargin = 0.5*d2r;  s.axis = ax;
    S = [S, s]; %#ok<AGROW>
end

% ---- PTP 連続 GO ベンチマーク ----
for is = 1:numel(speeds)
    for i = 1:nPtp
        s = emptyScenario(nAx);
        s.pattern = sprintf('ptp_v%02d', round(100*speeds(is)));
        s.type = 'ptp';  s.phase = 'benchmark';
        s.name = sprintf('%s_%03d', s.pattern, i);
        s.seed = c.seedBase + 200000 + 1000*is + i;
        s.duration = durPtp;  s.speedScale = speeds(is);  s.nWp = nWp;
        S = [S, s]; %#ok<AGROW>
    end
end

% ---- リミット接触（3 軸のみ。docs/phase3_5_report.md。既存シナリオの後ろに追加し、既存のシード・初期状態・split を変えない）----
%   励振モデルのバリアを目標軸だけリミットの外側（位置 1.10）へ移し、速度バリアを 0.80 にして機械ストッパへの衝突を取る。
%   contact_fall  : 全軸無トルク（微小 BLN）で自由落下し、重力方向のリミットに衝突（バリアは全軸、位置 1.10・速度 0.80）
%   contact_drive : 目標軸（axis, side）へ定トルク（0.3〜1.0 × 定格）で衝突・押し付け。他軸は重力保持＋微小 BLN
%   contact_bln   : 目標軸に大振幅 BLN（1.0 × 定格、5 Hz）＋バイアス（0.15 × 定格、side 向き）、他軸は重力保持＋微小 BLN
%   contact_bounce: 目標軸のリミットの 2° 手前から壁向きの初速（0.3〜0.6 × qdMax）で入る。J1 は壁向きのバイアス（+0.15）、
%                   J2・J3 は壁から離れる向きのバイアス（−0.3）で、反発・往復による動的な接触を増やす（事前の実測）
if nAx == 3 && strcmp(preset, 'full')
    kinds = {'contact_fall', 12, 3; 'contact_drive', 18, 3; 'contact_bln', 18, 6; 'contact_bounce', 12, 1.5};   % {pattern, 本数, 継続時間}
    for kd = 1:size(kinds,1)
        for i = 1:kinds{kd,2}
            s = emptyScenario(nAx);
            s.pattern = kinds{kd,1};  s.type = 'excitation';  s.phase = 3;
            s.name = sprintf('%s_%03d', s.pattern, i);
            s.seed = c.seedBase + 300000 + 1000*kd + i;
            rs = RandStream('twister','Seed',s.seed);
            s.duration = kinds{kd,3};  s.fMax = 5;
            s.axis = mod(i-1, nAx) + 1;  s.side = 1 - 2*(mod(floor((i-1)/nAx), 2) == 1);   % 軸を巡回し、次の周回で反対側
            q0 = qLo + rand(rs,1,nAx).*(qHi - qLo);
            s.qSoftFrac = 0.85*ones(1,nAx);  s.dqSoftFrac = 0.60*ones(1,nAx);
            s.qSoftFrac(s.axis) = 1.10;  s.dqSoftFrac(s.axis) = 0.80;
            switch s.pattern
                case 'contact_fall'
                    s.ampRel = 0.05;  s.axis = 0;  s.side = 0;
                    q0 = 0.6*sp.qMin + rand(rs,1,nAx).*0.6.*(sp.qMax - sp.qMin);          % [0.6 qMin, 0.6 qMax] の乱数姿勢
                    s.q0 = q0;  s.dq0 = (2*rand(rs,1,nAx) - 1) .* (0.5*dqI);
                    s.qSoftFrac(:) = 1.10;  s.dqSoftFrac(:) = 0.80;
                case 'contact_drive'
                    s.ampRel = 0.3 + 0.7*rand(rs);  s.q0 = q0;  s.dq0 = zeros(1,nAx);
                case 'contact_bln'
                    s.ampRel = 1.0;  s.ampRel2 = 0.15;  s.q0 = q0;  s.dq0 = (2*rand(rs,1,nAx) - 1).*dqI;
                case 'contact_bounce'
                    a = s.axis;  base = [0.2 -0.5 0.8];  q0 = base(1:nAx) + 0.2*(2*rand(rs,1,nAx) - 1);      % 他軸は基準姿勢のまわり ±0.2 rad
                    q0(a) = ternary(s.side > 0, sp.qMax(a), sp.qMin(a)) - s.side*2*d2r;
                    s.q0 = q0;  s.dq0 = zeros(1,nAx);  s.dq0(a) = s.side*(0.3 + 0.3*rand(rs))*sp.qdMax(a);
                    s.ampRel2 = ternary(a == 1, 0.15, -0.30);     % 壁向き（符号は side を掛ける）／J2・J3 は離れる向き
            end
            S = [S, s]; %#ok<AGROW>
        end
    end
end

% ---- train / val / test をパターンごとに 70/15/15 で分割（シードで決定的）----
pats = unique({S.pattern}, 'stable');
for ip = 1:numel(pats)
    idx = find(strcmp({S.pattern}, pats{ip}));
    n = numel(idx);
    rs = RandStream('twister','Seed',c.seedBase + ip);
    ord = idx(randperm(rs, n));
    nTr = ceil(0.7*n);  nVa = floor(0.15*n);
    if n < 3, nTr = n; nVa = 0; end                   % 件数が少ないパターンは train に寄せる
    for j = 1:n
        if strcmp(S(ord(j)).phase,'benchmark'), S(ord(j)).split = 'benchmark';
        elseif j <= nTr, S(ord(j)).split = 'train';
        elseif j <= nTr + nVa, S(ord(j)).split = 'val';
        else, S(ord(j)).split = 'test';
        end
    end
end
end

function s = emptyScenario(nAx)
s = struct('name','','pattern','','type','','phase',1,'split','','seed',0,'duration',0, ...
    'ampRel',0,'fMax',0,'q0',zeros(1,nAx),'dq0',zeros(1,nAx),'qSoftFrac',0.85,'dqSoftFrac',0.60, ...
    'posture',zeros(1,nAx),'speedScale',0,'nWp',0,'refAmp',0,'refMargin',0,'axis',0,'side',0,'ampRel2',0);
end

function y = ternary(c, a, b)
if c, y = a; else, y = b; end
end
