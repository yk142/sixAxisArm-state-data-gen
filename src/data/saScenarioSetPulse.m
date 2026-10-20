function S = saScenarioSetPulse(nAx, n1, nOther, grid)
%SASCENARIOSETPULSE 高トルク領域を埋めるパルス励振のシナリオ表（issue #67）。
%   S = SASCENARIOSETPULSE(nAx, n1, nOther, grid)
%   閉ループの高ゲインのステップ応答では、軸のトルクが上限（tauPeak）まで出るが、既存データには軸 1 のトルクが最大値の 3 割を超える行が 0.1% しかない。
%   pattern 'pulse'（type 'excitation'）: 静止状態から、目標軸に ±(0.3〜1)·tauPeak のトルクパルス（20〜150 ms）を 3 回（符号は交互、間隔 0.4 s）加える。
%   他軸は初期姿勢の重力保持＋微小 BLN。軸 1 を n1 本、軸 2〜nAx を各 nOther 本。ペイロード [kg] は grid を巡回して割り当てる。
%   split は pattern 'pulse' 内で 70/15/15（目標軸ごとに層別）。名前 pulse_a<軸>_<番号>、seed = seedBase + 300000 + 1000·軸 + 番号。
%   S(k).payload、S(k).replica（= 0）を持つ（saScenarioSetPayload の出力と連結できる）。
c = sa_config();  sp = saParams(nAx);  d2r = pi/180;
qLo = sp.qMin + (1-0.85)*(sp.qMax-sp.qMin)/2 + 2*d2r;  qHi = sp.qMax - (1-0.85)*(sp.qMax-sp.qMin)/2 - 2*d2r;
cnt = [n1, repmat(nOther, 1, nAx-1)];
U = saSobol(sum(cnt), nAx, c.seedBase + 777);
S = struct([]);  k = 0;
for a = 1:nAx
    for i = 1:cnt(a)
        k = k + 1;
        s = struct('name',sprintf('pulse_a%d_%03d',a,i),'pattern','pulse','type','excitation','phase',2,'split','train', ...
            'seed',c.seedBase + 300000 + 1000*a + i,'duration',1.4,'ampRel',0.05,'fMax',5,'q0',qLo + U(k,:).*(qHi - qLo),'dq0',zeros(1,nAx), ...
            'qSoftFrac',0.85,'dqSoftFrac',0.60,'posture',zeros(1,nAx),'speedScale',0,'nWp',0,'refAmp',0,'refMargin',0,'axis',a,'side',0,'ampRel2',0, ...
            'payload',grid(mod(k-1,numel(grid))+1),'replica',0);
        S = [S, s]; %#ok<AGROW>
    end
    idx = find([S.axis] == a);  n = numel(idx);                       % 目標軸ごとに 70/15/15
    rs = RandStream('twister','Seed',c.seedBase + 888 + a);  ord = idx(randperm(rs, n));
    nTr = ceil(0.7*n);  nVa = floor(0.15*n);
    for j = 1:n
        if j <= nTr, S(ord(j)).split = 'train'; elseif j <= nTr + nVa, S(ord(j)).split = 'val'; else, S(ord(j)).split = 'test'; end
    end
end
end
