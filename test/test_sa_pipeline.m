classdef test_sa_pipeline < matlab.unittest.TestCase
%TEST_SA_PIPELINE 1 kHz 変換・組み立て・検証の単体テスト（シミュレーションなし。issue #19）。
%   実行: results = runtests('test/test_sa_pipeline.m')

    properties
        S
        sp
    end

    methods (TestClassSetup)
        function paths(tc)
            repo = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
            S = saScenarioSet(3, 'small');
            tc.S = S(ismember({S.name}, {'bln_normal_001','bln_normal_002','bln_high_001','hold_p1_001','ptp_v30_001'}));
            tc.sp = saParams(3);
        end
    end

    methods (Access = private)
        function e = fakeCache(tc, s, K, seed)
            % 可動範囲の中央付近の滑らかな合成データ（q, dq は整合していなくてよい）
            rs = RandStream('twister','Seed',seed);  N = 8*K + 1;  t = (0:N-1)'/8000;
            mid = (tc.sp.qMax + tc.sp.qMin)/2;
            ph = 2*pi*rand(rs,1,3);
            q = mid + 0.1*sin(2*pi*(1:3).*t + ph);
            dq = 0.1*cos(2*pi*(1:3).*t + ph);
            tau = 0.1*tc.sp.tauPeak .* (2*rand(rs,N,3) - 1);
            e = struct('name',s.name,'seed',s.seed,'q',q,'dq',dq,'tau',tau,'qref',[],'dqref',[], ...
                'q0',q(1,:),'dq0',dq(1,:),'contact8k',0,'fs',8000,'wall',0);
        end
        function ds = assemble(tc, mod)
            if nargin < 2, mod = @(e, s) e; end
            L = containers.Map();
            for k = 1:numel(tc.S)
                s = tc.S(k);  L(s.name) = mod(tc.fakeCache(s, 40 + k, 100 + k), s);
            end
            ds = saAssembleDataset(tc.S, @(s) L(s.name), 3, struct('preset','small'));
        end
    end

    methods (Test)
        function downsampleTrapezoid(tc)
            rs = RandStream('twister','Seed',1);  K = 7;  N = 8*K + 1 + 3;      % 端数のサンプルは捨てる
            raw.q = rand(rs,N,3);  raw.dq = rand(rs,N,3);  raw.tau = rand(rs,N,3);
            D = saDownsample(raw, 8000, 1000);
            tc.verifyEqual(size(D.q), [K+1 3]);  tc.verifyEqual(size(D.tau), [K 3]);
            tc.verifyEqual(D.q, raw.q(1:8:8*K+1, :));  tc.verifyEqual(D.dq, raw.dq(1:8:8*K+1, :));
            for k = 1:K
                seg = raw.tau(8*(k-1)+1 : 8*k+1, :);            % 区間の 9 点（端点を含む）
                tc.verifyEqual(D.tau(k,:), trapz(seg, 1)/8, 'AbsTol', 1e-12);
            end
            tc.verifyEqual(D.tauInst, raw.tau(1:8:8*(K-1)+1, :));
        end

        function downsampleIsExactForPiecewiseLinear(tc)
            % FOH で補間された入力（8 kHz 格子の点を直線で結ぶ）の区間平均は、台形平均で厳密
            t = (0:8*5)'/8000;  raw.tau = [3*t, -2*t + 1, 0*t + 0.5];  raw.q = raw.tau;  raw.dq = raw.tau;
            D = saDownsample(raw, 8000, 1000);
            tmid = ((0:4)' + 0.5)/1000;
            tc.verifyEqual(D.tau, [3*tmid, -2*tmid + 1, 0*tmid + 0.5], 'AbsTol', 1e-12);
            % 区間先頭値との差は、勾配 × 区間長 / 2（台形平均が終点側を取り込んでいる）
            tc.verifyEqual(D.tau(:,1) - D.tauInst(:,1), 3*0.5e-3*ones(5,1), 'AbsTol', 1e-12);
        end

        function downsampleRejectsNonIntegerRatio(tc)
            raw.q = zeros(20,3);  raw.dq = raw.q;  raw.tau = raw.q;
            tc.verifyError(@() saDownsample(raw, 8000, 3000), 'saDownsample:ratio');
        end

        function assembleStatsUseTrainOnly(tc)
            ds = tc.assemble();
            trn = strcmp({ds.scen.split}, 'train');
            tc.assertTrue(any(trn) && any(~trn));
            X = [];  Y = [];
            for k = find(trn)
                sc = ds.scen(k);
                X = [X; sc.q(1:end-1,:), sc.dq(1:end-1,:), sc.tau];  Y = [Y; diff(sc.q), diff(sc.dq)]; %#ok<AGROW>
            end
            tc.verifyEqual(ds.stats.xMean, mean(X,1), 'AbsTol', 1e-12);
            tc.verifyEqual(ds.stats.xStd, std(X,0,1), 'RelTol', 1e-9);
            tc.verifyEqual(ds.stats.yStd, std(Y,0,1), 'RelTol', 1e-9);
            tc.verifyEqual(ds.stats.nTrain, size(X,1));
            % val / test / benchmark を変えても統計は変わらない（リーク防止）
            ds2 = tc.assemble(@(e, s) ternary(~strcmp(s.split,'train'), setfield(e, 'tau', 5*e.tau), e)); %#ok<SFLD>
            tc.verifyEqual(ds2.stats.xMean, ds.stats.xMean);  tc.verifyEqual(ds2.stats.yStd, ds.stats.yStd);
        end

        function assembleManifestAndShapes(tc)
            ds = tc.assemble();  m = ds.manifest;
            tc.verifyEqual(m.nAx, 3);  tc.verifyEqual(m.nScenario, numel(tc.S));
            tc.verifyEqual(m.nTransition, sum([ds.scen.nTransition]));
            tc.verifyEqual(m.nTrain + m.nVal + m.nTest + m.nBenchmark, m.nScenario);
            tc.verifyEqual(numel(m.scenarioHash), 64);
            tc.verifyEqual(size(ds.scen(1).q, 2), 3);
            tc.verifyEqual(numel(ds.scen(1).atLimit), ds.scen(1).nTransition);
        end

        function assembleRejectsMismatchedCache(tc)
            L = containers.Map();
            for k = 1:numel(tc.S), L(tc.S(k).name) = tc.fakeCache(tc.S(k), 10, k); end
            S2 = tc.S;  S2(1).seed = S2(1).seed + 1;
            tc.verifyError(@() saAssembleDataset(S2, @(s) L(s.name), 3), 'saAssembleDataset:cache');
        end

        function atLimitFlagsLimitTransitions(tc)
            ds = tc.assemble(@(e, s) putAtLimit(e, tc.sp));
            sc = ds.scen(1);
            tc.verifyTrue(any(sc.atLimit) && ~all(sc.atLimit));
            tc.verifyEqual(find(sc.atLimit)', [4 5]);        % q(5) がリミット上 → 遷移 4（終了側）と 5（開始側）
        end

        function repoStateAndProvenance(tc)
            r = saRepoState();
            tc.verifyEqual(numel(r.commit), 40);  tc.verifyClass(r.dirty, 'logical');
            if r.dirty, tc.verifyEqual(numel(r.diffHash), 16); else, tc.verifyEmpty(r.diffHash); end
            clean = struct('commit',repmat('a',1,40),'dirty',false,'diffHash','');
            dirty = struct('commit',repmat('a',1,40),'dirty',true,'diffHash','0123456789abcdef');
            % 全件がクリーンかつ同一コミット → genClean
            ds = tc.assemble(@(e, s) setfield(setfield(e,'repo',clean),'matlab','R-test')); %#ok<SFLD>
            tc.verifyTrue(ds.manifest.genClean);  tc.verifyEqual(numel(ds.manifest.genRepo), 1);
            tc.verifyEqual(ds.manifest.genMatlab, {'R-test'});
            % 1 本でも dirty → genClean = false、来歴は 2 種類並ぶ
            ds = tc.assemble(@(e, s) setfield(e,'repo',ternary(strcmp(s.name,'hold_p1_001'), dirty, clean))); %#ok<SFLD>
            tc.verifyFalse(ds.manifest.genClean);  tc.verifyEqual(numel(ds.manifest.genRepo), 2);
            % 来歴の無いキャッシュ → genClean = false
            tc.verifyFalse(tc.assemble().manifest.genClean);
            % 生成時間の合計
            ds = tc.assemble(@(e, s) setfield(e,'wall',2));  %#ok<SFLD>
            tc.verifyEqual(ds.manifest.wallTotal, 2*numel(tc.S));
        end

        function validatePassesAndDetectsFaults(tc)
            ds = tc.assemble();
            r = validateSaData(ds, struct('nSample',30,'verbose',false));
            tc.verifyTrue(r.ok, strjoin(r.fail, '; '));
            tc.verifyEqual(size(r.metrics.residRatio), [1 3]);
            % NaN
            d1 = ds;  d1.scen(1).q(3,1) = NaN;
            tc.verifyFalse(validateSaData(d1, struct('nSample',5,'verbose',false)).ok);
            % トルク超過
            d2 = ds;  d2.scen(2).tau(2,2) = 2*tc.sp.tauPeak(2);
            r2 = validateSaData(d2, struct('nSample',5,'verbose',false));
            tc.verifyFalse(r2.ok);  tc.verifyTrue(any(contains(r2.fail, 'tauPeak')));
            % リーク: PTP が train
            d3 = ds;  i = find(strcmp({d3.scen.type},'ptp'));  d3.scen(i).split = 'train';
            tc.verifyFalse(validateSaData(d3, struct('nSample',5,'verbose',false)).ok);
            % シード重複
            d4 = ds;  d4.scen(2).seed = d4.scen(1).seed;
            tc.verifyFalse(validateSaData(d4, struct('nSample',5,'verbose',false)).ok);
        end
    end
end

function y = ternary(c, a, b)
if c, y = a; else, y = b; end
end

function e = putAtLimit(e, sp)
e.q(8*4 + 1, 1) = sp.qMax(1);                % 遷移 4 の終了状態 = 5 番目の 1 kHz 点
end
