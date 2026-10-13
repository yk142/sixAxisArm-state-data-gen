classdef test_sa_scenarioset < matlab.unittest.TestCase
%TEST_SA_SCENARIOSET シナリオ表・Sobol 列・励振信号の単体テスト（シミュレーションなし。issue #15）。
%   実行: results = runtests('test/test_sa_scenarioset.m')

    methods (TestClassSetup)
        function paths(~)
            repo = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
        end
    end

    methods (Test)
        function fullCounts(tc)
            S = saScenarioSet(3, 'full');
            tc.verifyEqual(numel(S), 242);                    % 158 + リミット接触 84（issue #27）
            tc.verifyEqual(sum([S.duration]), 1542);          % ptp は継続時間の上限（30 s × 12 本 = 360 s を含む）
            pats = unique({S.pattern});
            for want = {'bln_normal','bln_high','bln_low','freefall','chirp','step','micro','nearlimit','ptp_v20','ptp_v80','hold_p1','hold_p6'}
                tc.verifyTrue(ismember(want{1}, pats), want{1});
            end
            tc.verifyEqual(sum(strcmp({S.type},'ptp')), 12);
            tc.verifyEqual(sum(strcmp({S.pattern},'nearlimit')), 12);
        end

        function sixAxisTable(tc)
            % 6 軸（issue #17）: 3 軸と同じパターン構成、状態は 6 軸、手首姿勢は p1〜p4 が特異点（J5=0）から離れ、p6 が J5=0
            S = saScenarioSet(6, 'full');  S3 = saScenarioSet(3, 'full');
            tc.verifyEqual(numel(unique({S.name})), numel(S));
            tc.verifyEqual(numel(unique([S.seed])), numel(S));
            tc.verifyEqual(size(S(1).q0), [1 6]);
            ex = strcmp({S.type},'excitation');
            tc.verifyEqual(sum(ex), sum(strcmp({S3.type},'excitation') & ~strncmp({S3.pattern},'contact_',8)));   % 接触は 3 軸のみ
            tc.verifyEqual(sum(strcmp({S.pattern},'nearlimit')), 24);       % 6 軸 x 両側 x 2 周
            tc.verifyEqual(sum(strcmp({S.type},'ptp')), 12);
            tc.verifyEqual(sum(strcmp({S.type},'hold')) - 24, sum(strcmp({S3.type},'hold')) - 12);   % nearlimit を除く hold は同数
            for ip = 1:6
                h = S(strcmp({S.name}, sprintf('hold_p%d_001', ip)));
                if ip <= 4, tc.verifyGreaterThanOrEqual(abs(h.posture(5)), 0.5, sprintf('p%d', ip));
                else, tc.verifyLessThanOrEqual(abs(h.posture(5)), 0.2, sprintf('p%d', ip)); end
            end
            tc.verifyEqual(S(strcmp({S.name},'hold_p6_001')).posture(5), 0);
            % 3 軸の姿勢は 6 軸の先頭 3 列と同じ
            tc.verifyEqual(S(strcmp({S.name},'hold_p2_001')).posture(1:3), S3(strcmp({S3.name},'hold_p2_001')).posture);
            % 初期状態はバリア帯の内側（6 軸）
            sp = saParams(6);  q0 = vertcat(S(ex).q0);
            lo = sp.qMin + 0.15*(sp.qMax-sp.qMin)/2 + deg2rad(2);  hi = sp.qMax - 0.15*(sp.qMax-sp.qMin)/2 - deg2rad(2);
            tc.verifyTrue(all(q0 >= lo - 1e-12, 'all') && all(q0 <= hi + 1e-12, 'all'));
            % nearlimit の姿勢: 対象軸がリミットの 4° 内側、他軸は J5 が特異点から離れている
            for s = S(strcmp({S.pattern},'nearlimit'))
                tc.verifyGreaterThan(abs(s.posture(5)), 0.5, s.name);
                if s.axis == 5, continue; end
                tc.verifyEqual(s.posture(5), 0.6, 'AbsTol', 1e-12, s.name);
            end
        end

        function namesAndSeedsAreUnique(tc)
            for preset = {'full','small'}
                S = saScenarioSet(3, preset{1});
                tc.verifyEqual(numel(unique({S.name})), numel(S));
                tc.verifyEqual(numel(unique([S.seed])), numel(S));
            end
        end

        function splitIsPerPatternAndDeterministic(tc)
            S1 = saScenarioSet(3, 'full');  S2 = saScenarioSet(3, 'full');
            tc.verifyEqual({S1.split}, {S2.split});
            tc.verifyEqual(unique({S1(strcmp({S1.type},'ptp')).split}), {'benchmark'});
            pats = unique({S1.pattern}, 'stable');
            for k = 1:numel(pats)
                sp_ = {S1(strcmp({S1.pattern}, pats{k})).split};
                if strcmp(pats{k}(1:3), 'ptp'), continue; end
                n = numel(sp_);
                if n >= 8      % 件数が十分（floor(0.15 n) >= 1）なら val・test にも入る
                    tc.verifyGreaterThan(sum(strcmp(sp_,'train')), 0, pats{k});
                    tc.verifyGreaterThan(sum(strcmp(sp_,'val')), 0, pats{k});
                    tc.verifyGreaterThan(sum(strcmp(sp_,'test')), 0, pats{k});
                end
            end
            nTr = sum(strcmp({S1.split},'train'));  nOther = sum(~strcmp({S1.split},'train') & ~strcmp({S1.split},'benchmark'));
            tc.verifyGreaterThan(nTr, nOther);
        end

        function initialStatesAreInsideBand(tc)
            sp = saParams(3);  S = saScenarioSet(3, 'full');  S = S(strcmp({S.type},'excitation') & ~strncmp({S.pattern},'contact_',8));   % 接触はバリアの外から始まる
            q0 = vertcat(S.q0);  dq0 = vertcat(S.dq0);
            lo = sp.qMin + 0.15*(sp.qMax-sp.qMin)/2 + deg2rad(2);  hi = sp.qMax - 0.15*(sp.qMax-sp.qMin)/2 - deg2rad(2);
            tc.verifyTrue(all(q0 >= lo - 1e-12, 'all') && all(q0 <= hi + 1e-12, 'all'));
            tc.verifyTrue(all(abs(dq0) <= 0.6*sp.qdMax*0.8 + 1e-12, 'all'));
            % 各軸で範囲の広い部分を覆う（Sobol）
            tc.verifyGreaterThan(min((max(q0,[],1) - min(q0,[],1))./(hi - lo)), 0.8);
        end

        function sobolIsLowDiscrepancy(tc)
            X = saSobol(64, 6);
            for j = 1:6      % 各次元の 1 次元周辺: 4 分割で厳密に等分（Sobol は 2^m 点で成り立つ）
                tc.verifyEqual(histcounts(X(:,j), 0:0.25:1), 16*ones(1,4), sprintf('dim %d', j));
            end
            tc.verifyEqual(histcounts2(X(:,1), X(:,2), 0:0.25:1, 0:0.25:1), 4*ones(4), '次元 1-2 の 4x4 分割');
            tc.verifyEqual(size(unique(X,'rows'),1), 64);
            Xs = saSobol(64, 6, 123);       % digital shift 後も 1 次元周辺は等分で、再現可能
            tc.verifyEqual(Xs, saSobol(64, 6, 123));
            for j = 1:6, tc.verifyEqual(histcounts(Xs(:,j), 0:0.25:1), 16*ones(1,4)); end
            tc.verifyEqual(size(saSobol(10, 12), 2), 12);
        end

        function signalsAreBounded(tc)
            t = (0:1/8000:2)';
            u = saSignal('bln', t, 10, struct('fMax',20,'seed',1));
            tc.verifyLessThanOrEqual(max(abs(u)), 10);  tc.verifyGreaterThan(max(abs(u)), 9);
            tc.verifyEqual(u, saSignal('bln', t, 10, struct('fMax',20,'seed',1)));
            tc.verifyNotEqual(u, saSignal('bln', t, 10, struct('fMax',20,'seed',2)));
            c = saSignal('chirp', t, 5, struct('f0',0.1,'f1',20));
            tc.verifyLessThanOrEqual(max(abs(c)), 5 + 1e-9);
            st = saSignal('step', t, 7, struct('tOn',0.2,'sign',-1));
            tc.verifyEqual([st(1) st(end)], [0 -7]);
        end
    end
end
