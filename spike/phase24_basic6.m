% Phase 2-4 手順 1: 6 軸で励振・閉ループが動くかの基礎確認
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
clear functions;
sp = saParams(6);  tmp = tempname;  mkdir(tmp);
mE = buildSaExcitation(6, struct('modelName','sa_excite6_sp','slxPath',fullfile(tmp,'sa_excite6_sp.slx')));
mC = buildSaClosedLoop(6, struct('modelName','sa_closed6_sp','slxPath',fullfile(tmp,'sa_closed6_sp.slx')));
% 励振: 各軸 ±tauPeak の乱数 (400 ステップ保持) 2 s
q0 = [0.2 -0.9 1.0 0.3 0.6 -0.3]';
t = (0:1/8000:2)';  K = ceil(numel(t)/400);
rs = RandStream('mt19937ar','Seed',5);  u = sp.tauPeak.*(2*rand(rs,K,6)-1);
tauEx = u(min(floor((0:numel(t)-1)'/400)+1,K),:);
tic; o = simSa(mE, t, tauEx, q0); fprintf('excite 2s: %.1f s wall\n', toc);
fprintf('finite=%d  inLimits=%d  maxdq/qdMax=%s\n', all(isfinite(o.q),'all'), ...
    all(o.q>sp.qMin & o.q<sp.qMax,'all'), mat2str(max(abs(o.dq))./sp.qdMax,3));
% 閉ループ PTP
WP = [q0'; 0.9 -0.4 1.6 -0.4 0.9 0.5; -0.3 -1.2 0.6 0.5 -0.7 -0.6];
T = saPtpSegTime(WP, 0.6);  t = (0:1/8000:sum(T)+0.3)';
tic; c = simSaClosed(mC, t, saPtpReference(t, WP, T), [], WP(1,:)'); fprintf('PTP %.1f s: %.1f s wall\n', t(end), toc);
fprintf('PTP max err [deg] per axis: %s\n', mat2str(max(rad2deg(abs(c.q-c.qRef))),3));
fprintf('PTP max tau/tauPeak: %s\n', mat2str(max(abs(c.tau))./sp.tauPeak,3));
close_system(mE,0); close_system(mC,0); rmdir(tmp,'s');
