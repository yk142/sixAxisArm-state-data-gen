classdef test_sa_elastic < matlab.unittest.TestCase
%TEST_SA_ELASTIC 関節弾性（issue #71）のパラメータと剛体極限の検証。
%   実行: results = runtests('test/test_sa_elastic.m')
    methods (TestClassSetup)
        function paths(~)
            repo = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
        end
    end
    methods (Test)
        function paramsGiveRequestedFrequency(tc)
            el = saElasticParams(6, struct('f',20,'zeta',0.05,'thetaMax',1));
            tc.verifyEqual(el.f, 20*ones(1,6), 'RelTol', 1e-9);
            tc.verifyEqual(el.fb, 'motor');
            el2 = saElasticParams(6, struct('f',20,'kScale',4,'thetaMax',1));
            tc.verifyEqual(el2.k, 4*el.k, 'RelTol', 1e-12);
            tc.verifyEqual(el2.f, 40*ones(1,6), 'RelTol', 1e-9);
        end
        function wristStiffnessIsFloored(tc)
            % 慣性の小さい手首側が柔らかくなりすぎないよう、定格トルクでのねじれ角が thetaMax 以下
            sp = saParams(6);  el = saElasticParams(6, struct('f',20));
            tc.verifyLessThanOrEqual(sp.tauRated./el.k, 0.02*(1 + 1e-9)*ones(1,6));
        end
        function stiffLimitMatchesRigid(tc)
            % 剛性を非常に大きくすると、リンク側の閉ループ応答が剛体のものに一致する（3 軸、軸 1 の 2° ステップ）
            c = sa_config();  dt = 1/c.fsSim;  nAx = 3;  sp = saParams(nAx);
            P0 = [0.0 -0.5 0.5];  tVec = (0:dt:0.8)';  nT = numel(tVec);  wn = 60;  z = 1;
            qr = repmat(P0, nT, 1);  qr(tVec >= 0.3, 1) = P0(1) + deg2rad(2);
            REF = [tVec, qr, zeros(nT,nAx), saFeedforward(tVec, qr, zeros(nT,nAx), zeros(nT,nAx), struct('mass',0))];
            w = tempname;  mkdir(w);  cl = onCleanup(@() rmdir(w,'s'));
            res = cell(1,2);
            for k = 1:2
                nm = sprintf('sa_test_el%d', k);  o = struct('modelName',nm,'slxPath',fullfile(w,[nm '.slx']));
                if k == 2, o.elastic = struct('kScale',1e3); end
                mdl = buildSaClosedLoop(nAx, o);
                set_param([mdl '/Kp'],'Gain',mat2str(sp.Mmean*wn^2,12));  set_param([mdl '/Kd'],'Gain',mat2str(2*z*sp.Mmean*wn,12));
                res{k} = simSaClosed(mdl, tVec, REF, [], P0');  close_system(mdl, 0);
            end
            tc.verifyLessThan(max(abs(res{2}.q - res{1}.q), [], 'all'), deg2rad(0.01));
            tc.verifyLessThan(max(abs(res{2}.qm - res{2}.q), [], 'all'), deg2rad(0.01));      % 剛体極限ではモータ側 = リンク側
        end
    end
end
