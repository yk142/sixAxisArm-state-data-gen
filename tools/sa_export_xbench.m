function sa_export_xbench(outFile, specs, wn, zeta, tol, unmodeled)
%SA_EXPORT_XBENCH 外挿ベンチマーク（X-bench、issue #107）の PTP を Simscape（モデル外の効果入り真値）で実行して HDF5 に書く。
%   sa_export_xbench('data/export/x_speed.h5', specs)                     既定 wn 60・ζ 1、RelTol 1e-7・AbsTol 1e-9、unmodeled = struct()
%   sa_export_xbench('data/export/x_wn40.h5', specs, 40, 1)
%   specs: 構造体配列 {name, speed, payload [kg], wp ('train' | 'outer'), seed}
%     speed  : PTP の速度倍率（saPtpSegTime の speedScale。学習は 0.3〜0.9、P-bench は 0.2〜0.8）
%     payload: 手先ペイロード [kg]（学習・P-bench は 0）。フィードフォワードはペイロード込み（c8FeedForward ＋ saFeedforward の差分）
%     wp     : 'train' = 到達点を学習と同じ範囲（p.qTrainMin〜qTrainMax）から、'outer' = 軸ごとに学習範囲の外側（学習範囲の端〜可動範囲の 5° 手前）から
%   その他は P-bench（saScenarioSet の ptp）と同じ: 到達点 4 点（nWp）、各到達点で 0.2 s 停留、5 次多項式。
%   形式は sa_export_ptp_el と同じ（ref、tau_d、truth_q / dq / qm / dqm を 8 kHz、controller）。属性 payload・speed・wp
%   実行: matlab -batch "addpath('tools'); sa_export_xbench('data/export/x_speed.h5', struct('name',{'xs_v10_1'},'speed',0.1,'payload',0,'wp','train','seed',1))"
repo = fileparts(fileparts(mfilename('fullpath')));
if ~startsWith(outFile, filesep), outFile = fullfile(pwd, outFile); end
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
if nargin < 3 || isempty(wn), wn = 60; end
if nargin < 4 || isempty(zeta), zeta = 1; end
if nargin < 5 || isempty(tol), tol = struct('RelTol',1e-7,'AbsTol',1e-9); end
if nargin < 6, unmodeled = struct(); end
c = sa_config();  dt = 1/c.fsSim;  p = c8_params(c.meshDir);  sp = saParams(6);  d2r = pi/180;
if exist(outFile,'file'), delete(outFile); end
pls = unique([specs.payload]);
for pm = pls
    workDir = tempname;  mkdir(workDir);  oldDir = cd(workDir);
    nm = sprintf('sa_xb_pl%d', round(10*pm));
    oc = struct('modelName',nm,'slxPath',fullfile(workDir,[nm '.slx']),'payload',struct('mass',pm),'elastic',struct(), ...
                'wn',wn,'zeta',zeta);
    if ~isempty(unmodeled), oc.unmodeled = unmodeled; end
    mdl = buildSaClosedLoop(6, oc);
    set_param(mdl,'RelTol',num2str(tol.RelTol,'%.12g'),'AbsTol',num2str(tol.AbsTol,'%.12g'),'MaxStep',num2str(dt,'%.12g'));
    fprintf('Simscape: payload %g kg、wn %g ζ %g、RelTol %g AbsTol %g\n', pm, wn, zeta, tol.RelTol, tol.AbsTol);
    for s = specs([specs.payload] == pm)
        t0 = tic;  g = ['/' s.name];
        rs = RandStream('twister','Seed',9500000 + s.seed);
        nWp = 4;  lo = p.qTrainMin(1:6);  hi = p.qTrainMax(1:6);
        switch s.wp
            case 'train'
                WP = lo + (hi - lo).*rand(rs, nWp, 6);
            case 'outer'                                   % 軸ごとに上側か下側の帯（学習範囲の端〜可動範囲の 5° 手前）
                side = rand(rs, nWp, 6) < 0.5;  u = rand(rs, nWp, 6);
                upLo = hi;  upHi = sp.qMax - 5*d2r;  dnLo = sp.qMin + 5*d2r;  dnHi = lo;
                WP = side.*(upLo + (upHi - upLo).*u) + (~side).*(dnLo + (dnHi - dnLo).*u);
            otherwise
                error('sa_export_xbench:wp', 'wp は train か outer: %s', s.wp);
        end
        T = saPtpSegTime(WP, s.speed);
        WP2 = WP(1,:);  T2 = [];
        for k = 1:numel(T)
            WP2(end+1,:) = WP(k+1,:);  T2(end+1,1) = T(k);      %#ok<AGROW>
            WP2(end+1,:) = WP(k+1,:);  T2(end+1,1) = 0.2;       %#ok<AGROW>
        end
        tVec = (0:dt:sum(T2))';
        REF = saPtpReference(tVec, WP2, T2);
        if pm ~= 0                                         % ペイロード分のフィードフォワードを足す（c8FeedForward はペイロードなし）
            qr = REF(:,2:7);  dqr = REF(:,8:13);  ddqr = gradient(dqr', dt)';
            REF(:,14:19) = REF(:,14:19) + saFeedforward(tVec, qr, dqr, ddqr, struct('mass',pm)) - saFeedforward(tVec, qr, dqr, ddqr);
        end
        tauD = zeros(numel(tVec), 6);
        o = simSaClosed(mdl, tVec, REF, tauD, REF(1,2:7)');
        w = @(k, v) wr(outFile, [g '/' k], v);
        w('ref', REF);  w('tau_d', tauD);  w('truth_q', o.q);  w('truth_dq', o.dq);  w('truth_qm', o.qm);  w('truth_dqm', o.dqm);  w('truth_tau', o.tau);
        h5writeatt(outFile, g, 'type', 'ptp');  h5writeatt(outFile, g, 'pattern', s.wp);  h5writeatt(outFile, g, 'split', 'xbench');
        h5writeatt(outFile, g, 'payload', pm);  h5writeatt(outFile, g, 'speed', s.speed);  h5writeatt(outFile, g, 'sec', toc(t0));
        fprintf('%s（%.1f s の動作）%.0f s\n', s.name, tVec(end), toc(t0));
    end
    close_system(mdl, 0);  cd(oldDir);
end
wr(outFile, '/controller/Mmean', sp.Mmean(:)');  wr(outFile, '/controller/tauPeak', sp.tauPeak(:)');
wr(outFile, '/controller/wn', wn);  wr(outFile, '/controller/zeta', zeta);
end

function wr(f, name, v)
h5create(f, name, size(v), 'Datatype', 'double');
h5write(f, name, double(v));
end
