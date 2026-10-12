classdef test_sa_closedloop < matlab.unittest.TestCase
%TEST_SA_CLOSEDLOOP 腕 3 軸の PD＋RNEA FF 閉ループモデルと PTP 参照軌道の検証（issue #13）。
%   実行: results = runtests('test/test_sa_closedloop.m')

    properties
        sp
        p
        mdl
        tmpDir
    end

    properties (Constant)
        wn = 60;
    end

    methods (TestClassSetup)
        function build(tc)
            repo = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
            tc.sp = saParams(3);
            tc.p = c8_params(sa_config().meshDir);
            tc.tmpDir = tempname;  mkdir(tc.tmpDir);
            tc.mdl = buildSaClosedLoop(3, struct('modelName','sa_closed3_test', ...
                'slxPath',fullfile(tc.tmpDir,'sa_closed3_test.slx')));
        end
    end

    methods (TestClassTeardown)
        function cleanup(tc)
            if bdIsLoaded(tc.mdl), close_system(tc.mdl,0); end
            if isfolder(tc.tmpDir), rmdir(tc.tmpDir,'s'); end
        end
    end

    methods (Access = private)
        function WP = randWaypoints(tc, seed, nWp)
            rs = RandStream('mt19937ar','Seed',seed);
            lo = 0.5*tc.sp.qMin;  hi = 0.5*tc.sp.qMax;
            WP = lo + (hi - lo).*rand(rs, nWp, 3);
        end
    end

    methods (Test)
        function interface(tc)
            tc.verifyEqual(get_param(tc.mdl,'Solver'), 'ode15s');
            ref = [tc.mdl '/ref'];
            tc.verifyEqual(str2double(get_param(ref,'PortDimensions')), 12);
            kp = str2num(get_param([tc.mdl '/Kp'],'Gain'));   %#ok<ST2NM>
            tc.verifyEqual(kp, tc.sp.Mmean*tc.wn^2, 'RelTol', 1e-9);
        end

        function feedforwardIsRnea(tc)
            % 姿勢保持の参照: q_ref 一定、dq_ref=0、τ_ff = 重力補償（c8RNEA）
            q = [0.3 -0.8 0.9];
            REF = saPtpReference((0:1/8000:0.2)', q, 0.1);
            z = zeros(6,1);  g = c8RNEA([q'; zeros(3,1)], z, z, tc.p);
            tc.verifyEqual(REF(end,2:4), q, 'AbsTol', 1e-12);
            tc.verifyEqual(REF(end,5:7), zeros(1,3), 'AbsTol', 1e-12);
            tc.verifyEqual(REF(end,8:10), g(1:3)', 'AbsTol', 1e-9);
        end

        function ptpTracking(tc)
            % 4 点の PTP（速度スケール 0.6）。追従誤差 < 0.005°、トルク飽和に達しない
            WP = [0.2 -0.9 1.0; 0.9 -0.4 1.6; -0.3 -1.2 0.6; 0.5 -0.7 1.2];
            T = saPtpSegTime(WP, 0.6);  t = (0:1/8000:sum(T)+0.3)';
            o = simSaClosed(tc.mdl, t, saPtpReference(t, WP, T), [], WP(1,:)');
            tc.verifyTrue(all(isfinite(o.q),'all'));
            tc.verifyLessThan(max(rad2deg(abs(o.q - o.qRef)),[],'all'), 0.005);
            tc.verifyLessThan(max(abs(o.tau)./tc.sp.tauPeak,[],'all'), 1);
        end

        function ptpFastRandomWaypoints(tc)
            % 速度スケール 0.8、乱数ウェイポイント 2 組（可動域の ±50% 内）。安定で、追従誤差 < 0.05°
            for seed = [1 2]
                WP = tc.randWaypoints(seed, 3);
                T = saPtpSegTime(WP, 0.8);  t = (0:1/8000:sum(T)+0.2)';
                o = simSaClosed(tc.mdl, t, saPtpReference(t, WP, T), [], WP(1,:)');
                tc.verifyTrue(all(isfinite(o.q),'all'), sprintf('seed %d', seed));
                tc.verifyLessThan(max(rad2deg(abs(o.q - o.qRef)),[],'all'), 0.05, sprintf('seed %d', seed));
            end
        end

        function holdWithDisturbance(tc)
            % 姿勢保持＋外乱（各軸 ±0.3 定格、100 Hz 更新の乱数を補間）。参照からの偏差が 外乱/Kp 以下
            postures = [0 -0.9 1.0; 0.8 -0.3 0.5; -1.5 -1.9 2.2];
            rs = RandStream('mt19937ar','Seed',4);
            for ip = 1:3
                q0 = postures(ip,:)';  t = (0:1/8000:1.5)';
                K = ceil(1.5*100) + 1;
                tauD = interp1((0:K-1)'/100, 0.3*tc.sp.tauRated.*(2*rand(rs,K,3) - 1), t);
                o = simSaClosed(tc.mdl, t, saPtpReference(t, q0', 0.1), tauD, q0);
                dev = max(abs(o.q - o.qRef),[],1);
                bound = max(abs(tauD),[],1) ./ (tc.sp.Mmean*tc.wn^2);
                tc.verifyLessThan(dev, bound, sprintf('姿勢 %d', ip));
                tc.verifyLessThan(max(abs(o.tau)./tc.sp.tauPeak,[],'all'), 1);
            end
        end
    end
end
