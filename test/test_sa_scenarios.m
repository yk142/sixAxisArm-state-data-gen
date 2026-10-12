classdef test_sa_scenarios < matlab.unittest.TestCase
%TEST_SA_SCENARIOS 各シナリオ種別を実際に 8 kHz で実行して検証する（3 軸、ゲート G2。issue #15）。
%   small preset の代表シナリオを実行する。実行時間は約 10 分（モデル生成 + シナリオごとの Simscape コンパイル）。

    properties
        sp
        S
        models
        tmpDir
    end

    properties (TestParameter)
        scenarioName = {'bln_normal_001','bln_high_001','bln_low_001','freefall_001','chirp_001', ...
            'step_001','step_002','step_003','micro_001','hold_p1_001','hold_p2_001', ...
            'nearlimit_j1p_001','nearlimit_j2m_001','nearlimit_j3p_001','ptp_v30_001','ptp_v70_001'};
    end

    methods (TestClassSetup)
        function build(tc)
            repo = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
            tc.sp = saParams(3);
            tc.S = saScenarioSet(3, 'small');
            tc.tmpDir = tempname;  mkdir(tc.tmpDir);
            mE = buildSaExcitation(3, struct('modelName','sa_excite3_scn','slxPath',fullfile(tc.tmpDir,'sa_excite3_scn.slx')));
            mC = buildSaClosedLoop(3, struct('modelName','sa_closed3_scn','slxPath',fullfile(tc.tmpDir,'sa_closed3_scn.slx')));
            tc.models = struct('excite',mE,'closed',mC);
        end
    end

    methods (TestClassTeardown)
        function cleanup(tc)
            for m = {tc.models.excite, tc.models.closed}
                if bdIsLoaded(m{1}), close_system(m{1},0); end
            end
            if isfolder(tc.tmpDir), rmdir(tc.tmpDir,'s'); end
        end
    end

    methods (Test)
        function runsAndRespectsLimits(tc, scenarioName)
            s = tc.S(strcmp({tc.S.name}, scenarioName));
            tc.assertNumElements(s, 1);
            raw = runSaScenario(s, tc.models);
            nAx = 3;  n = numel(raw.t);
            tc.verifyEqual(size(raw.q), [n nAx]);
            tc.verifyTrue(all(isfinite(raw.q),'all') && all(isfinite(raw.dq),'all') && all(isfinite(raw.tau),'all'));
            tc.verifyLessThanOrEqual(max(abs(raw.tau),[],1), tc.sp.tauPeak*(1 + 1e-9));
            tc.verifyTrue(all(raw.q > tc.sp.qMin - deg2rad(1), 'all') && all(raw.q < tc.sp.qMax + deg2rad(1), 'all'));
            switch s.type
                case 'excitation'
                    tc.verifyEqual(raw.t(end), s.duration, 'AbsTol', 1e-9);
                    tc.verifyEqual(raw.contact, 0);                 % ソフトバリアで止まり、リミットに接触しない
                    tc.verifyTrue(all(abs(raw.dq) < tc.sp.qdMax, 'all'));
                    tc.verifyEqual(raw.q(1,:), s.q0, 'AbsTol', 1e-9);
                case 'hold'
                    tc.verifyEqual(size(raw.qref), [n nAx]);
                    tc.verifyLessThan(max(rad2deg(abs(raw.q - raw.qref)),[],'all'), 5);   % 外乱による偏差
                    tc.verifyLessThan(raw.contact, 0.2);
                case 'ptp'
                    tc.verifyLessThan(max(rad2deg(abs(raw.q - raw.qref)),[],'all'), 0.05);
                    tc.verifyEqual(raw.contact, 0);
            end
        end

        function isReproducible(tc)
            s = tc.S(strcmp({tc.S.name}, 'bln_normal_001'));
            r1 = runSaScenario(s, tc.models);  r2 = runSaScenario(s, tc.models);
            tc.verifyEqual(r1.q, r2.q);  tc.verifyEqual(r1.tau, r2.tau);
        end
    end
end
