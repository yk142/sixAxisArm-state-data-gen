classdef test_sa_eval < matlab.unittest.TestCase
%TEST_SA_EVAL 再現性比較・カバレッジ指標の単体テスト（シミュレーションなし。issue #23）。
    methods (TestClassSetup)
        function paths(~)
            repo = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
        end
    end
    methods (Test)
        function reproSampleSelection(tc)
            S = saScenarioSet(3,'full');
            n = saReproSample(S,'sample');
            tc.verifyEqual(numel(n), 39);                         % 3 軸 full 242 本（接触 4 パターンを含む）。接触追加前の 158 本では 31 本だった
            tc.verifyEqual(numel(unique(n)), 39);
            tc.verifyEqual(numel(saReproSample(S,'all')), 242);
            tc.verifyTrue(all(ismember(unique({S(ismember({S.name},n)).pattern}), unique({S.pattern}))));
            tc.verifyEqual(numel(unique({S(ismember({S.name},n)).pattern})), numel(unique({S.pattern})));   % 全パターンを含む
        end
        function compareDetectsOneBitDifference(tc)
            d = tempname;  mkdir(fullfile(d,'a'));  mkdir(fullfile(d,'b'));  cleanup = onCleanup(@() rmdir(d,'s')); %#ok<NASGU>
            e = struct('q',rand(50,3),'dq',rand(50,3),'tau',rand(50,3),'qref',[],'dqref',[],'q0',rand(1,3),'dq0',rand(1,3),'contact8k',0);
            save(fullfile(d,'a','full3_x.mat'),'e');  save(fullfile(d,'b','full3_x.mat'),'e');
            r = saCompareCache(fullfile(d,'a'), fullfile(d,'b'), {'x'}, 'full', 3);
            tc.verifyTrue(r.identical);  tc.verifyEqual(r.firstDiffStep, 0);
            e.tau(7,2) = nextAfter(e.tau(7,2));  save(fullfile(d,'b','full3_x.mat'),'e');
            r = saCompareCache(fullfile(d,'a'), fullfile(d,'b'), {'x'}, 'full', 3);
            tc.verifyFalse(r.identical);  tc.verifyEqual(r.firstDiffStep, 7);  tc.verifyGreaterThan(r.maxAbsDiff, 0);
        end
        function coverageSeparatesOverlappingAndDisjointSets(tc)
            sp = saParams(3);  rng(1);  mid = (sp.qMax+sp.qMin)/2;  half = (sp.qMax-sp.qMin)/2;
            mk = @(split, lo, hi) struct('split',split,'q',mid + half.*(lo + (hi-lo)*rand(2001,3)), 'dq',sp.qdMax.*(lo + (hi-lo)*rand(2001,3)));
            ds.manifest = struct('nAx',3,'params',sp);
            ds.scen = [mk('train',-0.9,0.9) mk('train',-0.9,0.9) mk('train',-0.9,0.9) mk('val',-0.9,0.9) mk('test',-0.9,0.9) mk('benchmark',-0.9,0.9)];
            r = saCoverage(ds, struct('nBins',2,'minCount',1));          % 2 分割 = 64 セル、同じ分布なら全セルが埋まる
            tc.verifyEqual(r.grid.benchmark, 1);
            ds.scen(1:3) = [mk('train',0.5,0.9) mk('train',0.5,0.9) mk('train',0.5,0.9)];     % train が正の側だけ
            ds.scen(6) = mk('benchmark',-0.9,-0.5);                                           % benchmark は負の側だけ
            r2 = saCoverage(ds, struct('nBins',2,'minCount',1));
            tc.verifyEqual(r2.grid.benchmark, 0);
        end
    end
end

function y = nextAfter(x)
y = x + eps(x);
end
