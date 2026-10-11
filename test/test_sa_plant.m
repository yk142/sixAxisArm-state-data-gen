classdef test_sa_plant < matlab.unittest.TestCase
%TEST_SA_PLANT 腕 3 軸／6 軸プラントの物理検証（issue #5、ゲート G1）。
%   実行: results = runtests('test/test_sa_plant.m')
%   プラントは一時フォルダに 1 回だけ生成する（3 軸・6 軸）。

    properties
        p
        tmpDir
        mdl   % containers.Map: nAx -> モデル名
    end

    properties (TestParameter)
        nAx = {3, 6};
    end

    methods (TestClassSetup)
        function buildPlants(tc)
            repo = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
            c = sa_config();
            tc.p = c8_params(c.meshDir);
            tc.tmpDir = tempname;  mkdir(tc.tmpDir);
            tc.mdl = containers.Map('KeyType','double','ValueType','char');
            for n = [3 6]
                name = sprintf('sa_plant%d_test', n);
                tc.mdl(n) = buildSaPlant(n, struct('modelName',name, ...
                    'slxPath',fullfile(tc.tmpDir,[name '.slx'])));
            end
        end
    end

    methods (TestClassTeardown)
        function cleanup(tc)
            for n = [3 6]
                if tc.mdl.isKey(n) && bdIsLoaded(tc.mdl(n)), close_system(tc.mdl(n),0); end
            end
            if isfolder(tc.tmpDir), rmdir(tc.tmpDir,'s'); end
        end
    end

    methods (Access = private)
        function q = randPose(tc, n, rs)
            % 可動域の ±60% 内の一様乱数姿勢 [n x 1]
            lo = 0.6*tc.p.qMin(1:n)';  hi = 0.6*tc.p.qMax(1:n)';
            q = lo + (hi-lo).*rand(rs, n, 1);
        end
        function g = gravTorque(tc, q)
            n = numel(q);  z = zeros(6,1);
            g = c8RNEA([q(:); zeros(6-n,1)], z, z, tc.p);  g = g(1:n);
        end
    end

    methods (Test)
        function interface(tc, nAx)
            m = tc.mdl(nAx);
            tc.verifyEqual(numel(find_system(m,'MaskType','Revolute Joint')), nAx);
            tc.verifyEqual(get_param(m,'Solver'), 'ode15s');
            tc.verifyEqual(str2double(get_param(m,'MaxStep')), 1/8000, 'AbsTol', 1e-15);
            scb = find_system(m,'MaskType','Solver Configuration');
            tc.verifyEqual(str2double(get_param(scb{1},'FilteringTimeConstant')), 1e-5, 'RelTol', 1e-9);
            o = simSa(m, [0;0.01], zeros(1,nAx), zeros(nAx,1));
            tc.verifyEqual(size(o.q,2), nAx);
            tc.verifyEqual(size(o.dq,2), nAx);
        end

        function holdsWithGravityCompensation(tc, nAx)
            % RNEA の重力補償トルクで静止（Simscape の重力・慣性配置 = c8RNEA）
            rs = RandStream('mt19937ar','Seed',1);
            for k = 1:4
                q0 = tc.randPose(nAx, rs);
                o = simSa(tc.mdl(nAx), (0:1/8000:0.5)', tc.gravTorque(q0)', q0);
                tc.verifyLessThan(max(abs(rad2deg(o.q - q0')),[],'all'), 1e-3);   % deg
                tc.verifyLessThan(max(abs(o.dq),[],'all'), 1e-3);                 % rad/s
            end
        end

        function matchesForwardDynamics(tc, nAx)
            % 前進動力学（c8RNEA＋摩擦）との短窓の一致: 初期過渡 10 ms を除き、
            % プラントの t=10 ms の状態から解析モデルを積分して 0.2 s まで比較
            rs = RandStream('mt19937ar','Seed',11);
            for k = 1:4
                q0 = tc.randPose(nAx, rs);
                dq0 = (k > 2) * 0.5*(2*rand(rs,nAx,1) - 1);
                tau = tc.gravTorque(q0) + 0.3*tc.p.tauRated(1:nAx)'.*(2*rand(rs,nAx,1) - 1);
                e = saCompareFwdDyn(tc.mdl(nAx), tc.p, q0, dq0, tau, 0.2, 8000, 0.01);
                tc.verifyLessThan(max(e.dqW), 0.01);                  % rad/s
                tc.verifyLessThan(max(rad2deg(e.qW)), 0.1);           % deg
                tc.verifyLessThan(max(e.dq), 0.01);                   % t=0 から積分した参照とも一致（初期過渡なし）
            end
        end

        function matchesForwardDynamicsUnderTorqueChanges(tc, nAx)
            % トルクが 1 ステップ(0.125 ms)ごとに変わる乱数列（各軸 ±20% 定格）でも解析モデルと一致する
            % （RMI の自動フィルタによる約 1 ms の遅れがないこと。docs/phase1_report.md）。
            % 格子上の標本値を線形補間(FOH)で与え、参照も同じ補間で積分する
            rs = RandStream('mt19937ar','Seed',3);
            q0 = tc.randPose(nAx, rs);
            seq = tc.gravTorque(q0)' + 0.2*tc.p.tauRated(1:nAx).*(2*rand(rs,40,nAx) - 1);
            e = saComparePwc(tc.mdl(nAx), tc.p, q0, seq, 8000, 1, true);
            tc.verifyLessThan(max(e.dqRms), 2e-3);        % rad/s
            tc.verifyLessThan(max(e.dq), 5e-3);           % rad/s
        end

        function freeFallReachesJ2Limit(tc, nAx)
            % J2 = −90° から無トルク → 負リミット付近に到達。発散なし
            q0 = zeros(nAx,1);  q0(2) = -pi/2;  q0(3) = 0.9;
            o = simSa(tc.mdl(nAx), (0:1/8000:5)', zeros(1,nAx), q0);
            tc.verifyTrue(all(isfinite(o.q),'all') && all(isfinite(o.dq),'all'));
            tc.verifyLessThan(min(o.q(:,2)), tc.p.qMin(2) + deg2rad(2));     % 到達
            tc.verifyGreaterThan(min(o.q(:,2)), tc.p.qMin(2) - deg2rad(3));  % めり込み 3° 以内
        end

        function reachesEachAxisLimit(tc, nAx)
            % 各軸を限界 0.2 rad 手前から押し込む（他軸は重力補償）。押し込みは J1〜J3 が定格トルク、
            % 手首 J4〜J6 は定格の 10%（定格では限界到達前に 700 rad/s 超まで加速し、非現実的な領域になる）。
            % 限界に到達し、めり込みが 5° 以内、数値発散なし
            for i = 1:nAx
                for s = [-1 1]
                    lim = (s > 0)*tc.p.qMax(i) + (s < 0)*tc.p.qMin(i);
                    q0 = zeros(nAx,1);  q0(2) = -0.5;  q0(3) = 0.8;  q0(i) = lim - s*0.2;
                    tau = tc.gravTorque(q0)';  tau(i) = tau(i) + s*tc.p.tauRated(i)*(1 - 0.9*(i > 3));
                    o = simSa(tc.mdl(nAx), (0:1/8000:2)', tau, q0);
                    pen = s*(o.q(:,i) - lim);
                    tc.verifyTrue(all(isfinite(o.q),'all'), sprintf('J%d s=%+d', i, s));
                    tc.verifyGreaterThan(max(pen), -deg2rad(2), sprintf('J%d s=%+d 到達', i, s));
                    tc.verifyLessThan(max(pen), deg2rad(5), sprintf('J%d s=%+d めり込み', i, s));
                end
            end
        end
    end
end
