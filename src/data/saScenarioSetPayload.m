function S = saScenarioSetPayload(nAx, preset, nRep, grid)
%SASCENARIOSETPAYLOAD 手先ペイロードを変えた複製を加えたシナリオ表（issue #65）。
%   S = SASCENARIOSETPAYLOAD(nAx, preset, nRep, grid)
%   replica 0 = saScenarioSet そのまま（payload = 0 kg。既存のキャッシュを再利用できる）。
%   replica r = 1..nRep: ptp（benchmark）以外の各シナリオの複製。名前に _r<r>、seed に 1e6·r を足し、payload は grid [kg] を巡回して割り当てる。
%   励振の初期状態は、重複しないよう一様性を保つ変換（r = 1: 範囲の中点で反転、r ≥ 2: 範囲内で巡回シフト）で変える。split は元のシナリオと同じ。
%   S(k).payload [kg]、S(k).replica を持つ。
S = saScenarioSet(nAx, preset);
[S.payload] = deal(0);  [S.replica] = deal(0);
if nargin < 3 || nRep < 1, return; end
sp = saParams(nAx);  d2r = pi/180;
qLo = sp.qMin + (1-0.85)*(sp.qMax-sp.qMin)/2 + 2*d2r;  qHi = sp.qMax - (1-0.85)*(sp.qMax-sp.qMin)/2 - 2*d2r;
base = S(~strcmp({S.type},'ptp'));
T = S;  j = 0;
for r = 1:nRep
    for k = 1:numel(base)
        s = base(k);  j = j + 1;
        s.name = sprintf('%s_r%d', s.name, r);  s.seed = s.seed + 1000000*r;  s.replica = r;
        s.payload = grid(mod(j-1, numel(grid)) + 1);
        if strcmp(s.type,'excitation')
            if r == 1
                s.q0 = qLo + qHi - s.q0;  s.dq0 = -s.dq0;
            else
                w = qHi - qLo;  s.q0 = qLo + mod(s.q0 - qLo + (0.37*r)*w, w);
            end
        end
        T = [T, s]; %#ok<AGROW>
    end
end
S = T;
end
