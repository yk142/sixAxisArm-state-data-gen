classdef test_sa_excitation < matlab.unittest.TestCase
%TEST_SA_EXCITATION 腕 3 軸の開ループ励振モデル（軸ごとのバリア＋トルク飽和）の検証（issue #11）。
%   実行: results = runtests('test/test_sa_excitation.m')

    properties
        sp
        mdl
        tmpDir
    end

    methods (TestClassSetup)
        function build(tc)
            repo = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
            tc.sp = saParams(3);
            tc.tmpDir = tempname;  mkdir(tc.tmpDir);
            tc.mdl = buildSaExcitation(3, struct('modelName','sa_excite3_test', ...
                'slxPath',fullfile(tc.tmpDir,'sa_excite3_test.slx')));
        end
    end

    methods (TestClassTeardown)
        function cleanup(tc)
            if bdIsLoaded(tc.mdl), close_system(tc.mdl,0); end
            if isfolder(tc.tmpDir), rmdir(tc.tmpDir,'s'); end
        end
    end

    methods (Access = private)
        function tauEx = randomHold(tc, seed, T, holdSteps)
            % 各軸 ±tauPeak の一様乱数を holdSteps ステップ（8 kHz）ごとに更新
            t = (0:1/8000:T)';  K = ceil(numel(t)/holdSteps);
            rs = RandStream('mt19937ar','Seed',seed);
            u = tc.sp.tauPeak.*(2*rand(rs,K,3) - 1);
            tauEx = u(min(floor((0:numel(t)-1)'/holdSteps)+1, K), :);
        end
        function g = gravTorque(tc, q)
            z = zeros(6,1);  p = c8_params(sa_config().meshDir);
            g = c8RNEA([q(:); zeros(3,1)], z, z, p);  g = g(1:3)';
        end
    end

    methods (Test)
        function parameters(tc)
            sp = tc.sp;
            tc.verifyEqual(sp.nAx, 3);
            tc.verifyTrue(all(sp.Mmin <= sp.Mmean) && all(sp.Mmean <= sp.Mmax));
            tc.verifyTrue(all(sp.Mmin > 0));
            tc.verifyEqual(sp.tauPeak, 3*sp.tauRated, 'RelTol', 1e-12);
        end

        function barrierPositionIsParameter(tc)
            % 既定は位置 0.85・速度 0.60。リミットの外側（1.1）へ移せる
            [lo, hi, dq] = setSaBarrier(tc.mdl, tc.sp, 0.85, 0.60);
            mid = (tc.sp.qMax + tc.sp.qMin)/2;  half = (tc.sp.qMax - tc.sp.qMin)/2;
            tc.verifyEqual(lo, mid - 0.85*half, 'AbsTol', 1e-12);
            tc.verifyEqual(hi, mid + 0.85*half, 'AbsTol', 1e-12);
            tc.verifyEqual(dq, 0.60*tc.sp.qdMax, 'AbsTol', 1e-12);
            [lo2, hi2] = setSaBarrier(tc.mdl, tc.sp, 1.1, 0.80);
            tc.verifyTrue(all(lo2 < tc.sp.qMin) && all(hi2 > tc.sp.qMax));
            tc.verifyEqual(str2num(get_param([tc.mdl '/qDead'],'LowerValue')), lo2, 'AbsTol', 1e-9); %#ok<ST2NM>
            setSaBarrier(tc.mdl, tc.sp, 0.85, 0.60);      % 既定に戻す
        end

        function insideBandIsTransparent(tc)
            % 安全帯の内側では、励振トルクがそのまま印加される（励振の分布を歪めない）
            q0 = [0.2; -0.9; 1.0];
            t = (0:1/8000:0.3)';  rs = RandStream('mt19937ar','Seed',2);
            tauEx = tc.gravTorque(q0) + 0.03*tc.sp.tauRated.*(2*rand(rs,1,3) - 1) + 0*t;
            o = simSa(tc.mdl, t, tauEx, q0);
            mid = (tc.sp.qMax + tc.sp.qMin)/2;  half = 0.85*(tc.sp.qMax - tc.sp.qMin)/2;
            tc.verifyTrue(all(o.q > mid - half, 'all') && all(o.q < mid + half, 'all'));
            tc.verifyLessThan(max(abs(o.tau - interp1(t, tauEx, o.t)), [], 'all'), 1e-9);   % N*m
        end

        function strongExcitationRespectsLimits(tc)
            % 各軸 ±tauPeak（飽和）の乱数トルクを 2 s。可動域・最大速度・最大トルクを守り、発散しない
            q0 = [0.2; -0.9; 1.0];
            for seed = [5 6]
                tauEx = tc.randomHold(seed, 2, 400);
                o = simSa(tc.mdl, (0:1/8000:2)', tauEx, q0);
                tc.verifyTrue(all(isfinite(o.q),'all') && all(isfinite(o.dq),'all'));
                tc.verifyTrue(all(o.q > tc.sp.qMin, 'all') && all(o.q < tc.sp.qMax, 'all'), sprintf('seed %d 可動域', seed));
                tc.verifyTrue(all(abs(o.dq) < tc.sp.qdMax, 'all'), sprintf('seed %d 最大速度', seed));
                tc.verifyLessThanOrEqual(max(abs(o.tau),[],1), tc.sp.tauPeak*(1 + 1e-9));
            end
        end

        function freeFallStaysInsideBand(tc)
            % 無トルクの自由落下でも、ソフトバリアで止まり、リミットに接触しない
            q0 = [0; -pi/2; 0.9];
            o = simSa(tc.mdl, (0:1/8000:3)', zeros(1,3), q0);
            tc.verifyTrue(all(isfinite(o.q),'all'));
            tc.verifyTrue(all(o.q > tc.sp.qMin, 'all') && all(o.q < tc.sp.qMax, 'all'));
            tc.verifyTrue(all(abs(o.dq) < tc.sp.qdMax, 'all'));
        end
    end
end
