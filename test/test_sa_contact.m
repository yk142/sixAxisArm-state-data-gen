classdef test_sa_contact < matlab.unittest.TestCase
%TEST_SA_CONTACT リミット接触シナリオ（3 軸、issue #27）。表の検証（シミュレーションなし）と、代表 6 本の実行。
    properties
        sp
        S
        models
        tmpDir
    end
    properties (TestParameter)
        scenarioName = {'contact_fall_001','contact_drive_002','contact_drive_004','contact_bln_003','contact_bounce_001','contact_bounce_002','contact_bounce_006'};
    end
    methods (TestClassSetup)
        function build(tc)
            repo = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
            tc.sp = saParams(3);  tc.S = saScenarioSet(3, 'full');
        end
    end
    methods (Test)
        function tableCounts(tc)
            c = tc.S(strncmp({tc.S.pattern}, 'contact_', 8));
            tc.verifyEqual(numel(c), 60);
            tc.verifyEqual(cellfun(@(p) sum(strcmp({c.pattern}, p)), {'contact_fall','contact_drive','contact_bln','contact_bounce'}), [12 18 18 12]);
            for p = {'contact_drive','contact_bln','contact_bounce'}
                m = c(strcmp({c.pattern}, p{1}));
                for a = 1:3, for sd = [-1 1]
                    tc.verifyGreaterThanOrEqual(sum([m.axis] == a & [m.side] == sd), 2, sprintf('%s axis %d side %+d', p{1}, a, sd));
                end, end
            end
            tc.verifyEqual(numel(unique({tc.S.name})), numel(tc.S));
            tc.verifyEqual(numel(unique([tc.S.seed])), numel(tc.S));
            tc.verifyTrue(all(strcmp({c.type}, 'excitation')));
        end
        function existingScenariosAreUnchanged(tc)
            % 既存 158 本（名前・シード・split）は、生成済みデータセットのマニフェストと一致する（キャッシュが有効なまま）
            repo = fileparts(fileparts(mfilename('fullpath')));
            f = fullfile(repo, 'data', 'sa_dataset_full3.mat');
            tc.assumeTrue(isfile(f), 'sa_dataset_full3.mat が無いため省略');
            m = load(f, 'manifest');
            first = tc.S(1:158);
            tc.verifyEqual({first.name}, m.manifest.names);
            tc.verifyEqual([first.seed], m.manifest.seeds);
            tc.verifyEqual(first(1).name, 'bln_normal_001');
            tc.verifyTrue(all(~strncmp({first.pattern}, 'contact_', 8)));
        end
        function barrierIsOutsideOnlyOnTargetAxis(tc)
            for s = tc.S(strncmp({tc.S.pattern}, 'contact_', 8))
                if strcmp(s.pattern, 'contact_fall'), tc.verifyTrue(all(s.qSoftFrac == 1.10)); continue; end
                tc.verifyEqual(find(s.qSoftFrac > 1), s.axis, s.name);
                tc.verifyEqual(find(s.dqSoftFrac > 0.7), s.axis, s.name);
            end
        end
        function bounceStartsInsideWallWithSpeedTowardIt(tc)
            for s = tc.S(strcmp({tc.S.pattern}, 'contact_bounce'))
                a = s.axis;
                lim = tc.sp.qMax(a); if s.side < 0, lim = tc.sp.qMin(a); end
                tc.verifyEqual(abs(s.q0(a) - lim), deg2rad(2), 'AbsTol', 1e-12, s.name);
                tc.verifyGreaterThanOrEqual(s.side*s.dq0(a), 0.3*tc.sp.qdMax(a) - 1e-12, s.name);
                tc.verifyLessThanOrEqual(s.side*s.dq0(a), 0.6*tc.sp.qdMax(a) + 1e-12, s.name);
            end
        end
    end
    methods (Test, TestTags = {'sim'})
        function runsAndTouchesTheLimit(tc, scenarioName)
            tc.buildModelOnce();
            s = tc.S(strcmp({tc.S.name}, scenarioName));
            raw = runSaScenario(s, tc.models);
            tc.verifyTrue(all(isfinite(raw.q),'all') && all(isfinite(raw.dq),'all') && all(isfinite(raw.tau),'all'));
            tc.verifyLessThanOrEqual(max(abs(raw.tau),[],1), tc.sp.tauPeak*(1 + 1e-9));
            tc.verifyTrue(all(raw.q > tc.sp.qMin - deg2rad(1.5), 'all') && all(raw.q < tc.sp.qMax + deg2rad(1.5), 'all'), 'めり込みが 1.5° 以内');
            tc.verifyLessThanOrEqual(max(abs(raw.dq),[],1), 1.3*tc.sp.qdMax, '衝突の衝撃で一時的に qdMax を超えうる（1.3 倍まで）');
            tc.verifyGreaterThan(raw.contact, 0, 'リミットに到達する');
            tc.verifyLessThan(raw.contact, 0.95);
            if s.axis > 0     % 目標軸の接触、他軸は接触しない
                near = raw.q <= tc.sp.qMin + deg2rad(0.1) | raw.q >= tc.sp.qMax - deg2rad(0.1);
                tc.verifyGreaterThan(mean(near(:,s.axis)), 0, '目標軸が接触');
                other = setdiff(1:3, s.axis);
                tc.verifyEqual(mean(near(:,other),'all'), 0, '他軸は接触しない');
            end
        end
    end
    methods
        function buildModelOnce(tc)
            if ~isempty(tc.models), return; end
            tc.tmpDir = tempname;  mkdir(tc.tmpDir);
            mE = buildSaExcitation(3, struct('modelName','sa_excite3_ct','slxPath',fullfile(tc.tmpDir,'sa_excite3_ct.slx')));
            tc.models = struct('excite',mE,'closed','');
            tc.addTeardown(@() tc.cleanup());
        end
        function cleanup(tc)
            if bdIsLoaded(tc.models.excite), close_system(tc.models.excite,0); end
            if isfolder(tc.tmpDir), rmdir(tc.tmpDir,'s'); end
        end
    end
end
