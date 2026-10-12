classdef test_sa_gen < matlab.unittest.TestCase
%TEST_SA_GEN 生成パイプライン（shard 分担・キャッシュ再開・組み立て・検証）の結合テスト（3 軸、実シミュレーション。issue #19）。
%   small プリセットの 3 本（step_001, freefall_001, hold_p1_001）で通す。実行時間は約 10 分。
%   実行: results = runtests('test/test_sa_gen.m')

    properties
        outDir
        names = {'freefall_001','step_001','hold_p1_001'}
    end

    methods (TestClassSetup)
        function setup(tc)
            repo = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
            tc.outDir = tempname;  mkdir(tc.outDir);
        end
    end

    methods (TestClassTeardown)
        function cleanup(tc)
            if isfolder(tc.outDir), rmdir(tc.outDir,'s'); end
        end
    end

    methods (Access = private)
        function opts = gopts(tc, varargin)
            opts = struct('nAx',3,'preset','small','names',{tc.names},'outDir',tc.outDir);
            for i = 1:2:numel(varargin), opts.(varargin{i}) = varargin{i+1}; end
        end
        function fn = cacheFile(tc, name)
            fn = fullfile(tc.outDir,'cache',sprintf('small3_%s.mat', name));
        end
    end

    methods (Test, TestTags = {'ordered'})
        function a_shardsAreDisjointAndResumable(tc)
            S = saScenarioSet(3,'small');  [~, loc] = ismember(tc.names, {S.name});  [~, ord] = sort(loc);
            nm = tc.names(ord);                                   % 表の順（担当は表の順の位置で決まる）
            ds = genSaDataset(tc.gopts('shard',[1 2],'cacheOnly',true));
            tc.verifyEmpty(ds);
            tc.verifyTrue(isfile(tc.cacheFile(nm{1})) && ~isfile(tc.cacheFile(nm{2})) && isfile(tc.cacheFile(nm{3})));
            ds = genSaDataset(tc.gopts('shard',[2 2]));            % 残りを生成し、そのまま組み立ては保留（未生成なし → 組み立て）
            tc.verifyNotEmpty(ds);
            tc.verifyEqual(ds.manifest.nScenario, 3);
            tc.verifyEmpty(dir(fullfile(tc.outDir,'cache','*.tmp')));
        end

        function b_resumeReproducesBitIdentical(tc)
            nm = tc.names{2};  fn = tc.cacheFile(nm);
            C0 = load(fn);
            delete(fn);
            genSaDataset(tc.gopts('cacheOnly',true));              % 欠けた 1 本だけ再生成（別モデル構築）
            C1 = load(fn);
            tc.verifyEqual(C1.e.q, C0.e.q);  tc.verifyEqual(C1.e.dq, C0.e.dq);  tc.verifyEqual(C1.e.tau, C0.e.tau);
        end

        function c_assembledDataIsValidAndConsistent(tc)
            tStart = tic;
            ds = genSaDataset(tc.gopts());                         % 生成済み → シミュレーションなしで組み立てのみ
            tc.verifyLessThan(toc(tStart), 20);
            tc.verifyTrue(isfile(fullfile(tc.outDir,'sa_dataset_small3_subset.mat')));
            r = validateSaData(ds, struct('nSample',300));
            tc.verifyTrue(r.ok, strjoin(r.fail, '; '));
            tc.verifyTrue(all(isfinite(r.metrics.residRatio)));
            tc.verifyLessThan(max(r.metrics.residRatio), 0.5);     % 1 ステップは解析モデルでほぼ説明できる（報告値は発見に記録）
            % 1 kHz の状態は 8 kHz の瞬時値の間引き
            for k = 1:numel(ds.scen)
                C = load(tc.cacheFile(ds.scen(k).name));
                tc.verifyEqual(ds.scen(k).q, C.e.q(1:8:8*ds.scen(k).nTransition+1, :));
            end
        end
    end
end
