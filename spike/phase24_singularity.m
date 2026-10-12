% Phase 2-4 手順 4: 手首特異点（J5=0）近傍の閉ループと wn への感度（6 軸）
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));  clear functions;
sp = saParams(6);  p = c8_params(sa_config().meshDir);  tmp = tempname;  mkdir(tmp);
% 質量行列 J4-J6 ブロックの条件数 vs J5
z = zeros(6,1);  fprintf('J5 [rad]  cond(M_wrist)  cond(M_full)\n');
for q5 = [0 0.05 0.2 0.5 0.9]
    q = [0 -0.5 0.5 0.3 q5 -0.2]';  g = c8RNEA(q,z,z,p);  M = zeros(6);
    for j = 1:6, e = z; e(j) = 1; M(:,j) = c8RNEA(q,z,e,p) - g; end
    fprintf('%6.2f   %10.3g   %10.3g\n', q5, cond(M(4:6,4:6)), cond(M));
end
% PTP が J5=0 を通る（J5: +0.8 -> -0.8 -> +0.8）、速度 0.8
WP = [0.2 -0.9 1.0 0.3  0.8 -0.3;
      0.6 -0.5 1.3 -0.4 -0.8  0.5;
      0.1 -1.0 0.9 0.3  0.8 -0.2];
T = saPtpSegTime(WP, 0.8);  t = (0:1/8000:sum(T)+0.3)';  REF = saPtpReference(t, WP, T);
rs = RandStream('mt19937ar','Seed',4);  K = ceil(t(end)*100)+1;
tauD = interp1((0:K-1)'/100, 0.3*sp.tauRated.*(2*rand(rs,K,6)-1), t);
fprintf('\nwn   | PTP(J5<0を通過) 最大誤差 [deg] J1..J6      | max tau/tauPeak | 保持 p6(J5=0)+外乱 偏差 [deg] J1..J6\n');
for wn = [30 60 100]
    nm = sprintf('sa_closed6_wn%d',wn);
    m = buildSaClosedLoop(6, struct('modelName',nm,'slxPath',fullfile(tmp,[nm '.slx']),'wn',wn));
    o = simSaClosed(m, t, REF, [], WP(1,:)');
    t2 = (0:1/8000:min(1.5,t(end)))';  q0 = [0 -2.2 1.0 0 0 0]';  tauD2 = interp1(t, tauD, t2);
    h = simSaClosed(m, t2, saPtpReference(t2, q0', 0.1), tauD2, q0);
    fprintf('%4d | %s | %.3f | %s\n', wn, mat2str(max(rad2deg(abs(o.q-o.qRef))),3), max(abs(o.tau)./sp.tauPeak,[],'all'), mat2str(max(rad2deg(abs(h.q-h.qRef))),3));
    close_system(m,0);
end
rmdir(tmp,'s');
