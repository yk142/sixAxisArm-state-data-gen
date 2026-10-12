classdef test_sa_closedloop < matlab.unittest.TestCase
%TEST_SA_CLOSEDLOOP PD＋RNEA FF 閉ループモデルと PTP 参照軌道の検証（3 軸: issue #13、6 軸: issue #17）。
%   実行: results = runtests('test/test_sa_closedloop.m')

    properties (ClassSetupParameter)
        nAx = {3, 6};
    end

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
        function build(tc, nAx)
            repo = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
            tc.sp = saParams(nAx);
            tc.p = c8_params(sa_config().meshDir);
            tc.tmpDir = tempname;  mkdir(tc.tmpDir);
            tc.mdl = buildSaClosedLoop(nAx, struct('modelName',sprintf('sa_closed%d_test',nAx), ...
                'slxPath',fullfile(tc.tmpDir,sprintf('sa_closed%d_test.slx',nAx))));
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
            WP = lo + (hi - lo).*rand(rs, nWp, tc.sp.nAx);
        end
    end

    methods (Test)
        function interface(tc)
            tc.verifyEqual(get_param(tc.mdl,'Solver'), 'ode15s');
            ref = [tc.mdl '/ref'];
            tc.verifyEqual(str2double(get_param(ref,'PortDimensions')), 4*tc.sp.nAx);
            kp = str2num(get_param([tc.mdl '/Kp'],'Gain'));   %#ok<ST2NM>
            tc.verifyEqual(kp, tc.sp.Mmean*tc.wn^2, 'RelTol', 1e-9);
        end

        function feedforwardIsRnea(tc)
            % 姿勢保持の参照: q_ref 一定、dq_ref=0、τ_ff = 重力補償（c8RNEA）
            n = tc.sp.nAx;  q6 = [0.3 -0.8 0.9 0.4 0.7 -0.2];  q = q6(1:n);
            REF = saPtpReference((0:1/8000:0.2)', q, 0.1);
            z = zeros(6,1);  g = c8RNEA([q'; zeros(6-n,1)], z, z, tc.p);
            tc.verifyEqual(REF(end,2:1+n), q, 'AbsTol', 1e-12);
            tc.verifyEqual(REF(end,2+n:1+2*n), zeros(1,n), 'AbsTol', 1e-12);
            tc.verifyEqual(REF(end,2+2*n:1+3*n), g(1:n)', 'AbsTol', 1e-9);
        end

        function ptpTracking(tc)
            % 4 点の PTP（速度スケール 0.6）。追従誤差 < 0.005°、トルク飽和に達しない
            WP = [0.2 -0.9 1.0 0.3 0.6 -0.3; 0.9 -0.4 1.6 -0.4 0.9 0.5; -0.3 -1.2 0.6 0.5 -0.7 -0.6; 0.5 -0.7 1.2 -0.2 0.4 0.3];
            WP = WP(:,1:tc.sp.nAx);
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
            postures = [0 -0.9 1.0 0.3 0.6 -0.3; 0.8 -0.3 0.5 -0.5 -0.9 0.4; -1.5 -1.9 2.2 0.4 1.1 0.2];
            postures = postures(:,1:tc.sp.nAx);
            rs = RandStream('mt19937ar','Seed',4);
            for ip = 1:3
                q0 = postures(ip,:)';  t = (0:1/8000:1.5)';
                K = ceil(1.5*100) + 1;
                tauD = interp1((0:K-1)'/100, 0.3*tc.sp.tauRated.*(2*rand(rs,K,tc.sp.nAx) - 1), t);
                o = simSaClosed(tc.mdl, t, saPtpReference(t, q0', 0.1), tauD, q0);
                dev = max(abs(o.q - o.qRef),[],1);
                bound = max(abs(tauD),[],1) ./ (tc.sp.Mmean*tc.wn^2);
                tc.verifyLessThan(dev, bound, sprintf('姿勢 %d', ip));
                tc.verifyLessThan(max(abs(o.tau)./tc.sp.tauPeak,[],'all'), 1);
            end
        end
    end
end
