classdef test_sa_payload < matlab.unittest.TestCase
%TEST_SA_PAYLOAD 手先ペイロード合成の単体テスト（issue #61）。
%   実行: results = runtests('test/test_sa_payload.m')
    methods (TestClassSetup)
        function paths(~)
            repo = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
        end
    end
    methods (Test)
        function zeroPayloadIsIdentity(tc)
            p = c8_params(sa_config().meshDir);
            tc.verifyEqual(saApplyPayload(p, struct('mass',0)).links(7), p.links(7));
            tc.verifyEqual(saApplyPayload(p, []).links(7), p.links(7));
        end
        function massCenterAndInertiaAreConsistent(tc)
            p = c8_params(sa_config().meshDir);  L = p.links(7);
            pl = struct('mass',3,'offset',[0.01; -0.02; 0.05]);
            q = saApplyPayload(p, pl).links(7);
            tc.verifyEqual(q.mass, L.mass + 3, 'AbsTol', 1e-12);
            tc.verifyEqual(q.mass*q.com(:), L.mass*L.com(:) + 3*pl.offset, 'AbsTol', 1e-12);     % 一次モーメント
            % リンク座標系の原点まわりの慣性（= 重心まわり + m(|c|²E − ccᵀ)）が、元のリンク + 点質量に一致する
            about0 = @(m, c, I) I + m*((c(:)'*c(:))*eye(3) - c(:)*c(:)');
            tc.verifyEqual(about0(q.mass, q.com, q.inertiaC), about0(L.mass, L.com, L.inertiaC) + 3*((pl.offset'*pl.offset)*eye(3) - pl.offset*pl.offset'), 'AbsTol', 1e-12);
            tc.verifyEqual(q.inertiaC, q.inertiaC', 'AbsTol', 1e-15);
        end
        function otherLinksUnchanged(tc)
            p = c8_params(sa_config().meshDir);
            q = saApplyPayload(p, struct('mass',5));
            for k = [1:6]
                tc.verifyEqual(q.links(k), p.links(k));
            end
        end
        function gravityTorqueIncreasesAtShoulder(tc)
            % 水平に伸ばした姿勢で、ペイロードの重力モーメントだけ J2 の保持トルクが増える（符号は問わず大きさで確認）
            p = c8_params(sa_config().meshDir);  q = zeros(6,1);  z = zeros(6,1);
            g0 = c8RNEA(q, z, z, p);  g1 = c8RNEA(q, z, z, saApplyPayload(p, struct('mass',3)));
            tc.verifyGreaterThan(abs(g1(2) - g0(2)), 0.1);
            tc.verifyEqual(g1(1), g0(1), 'AbsTol', 1e-9);       % J1 は鉛直軸まわり（重力トルクなし）
        end
    end
end
