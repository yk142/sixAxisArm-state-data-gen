function r = sa_bench_simscape(outFile, elastic, nRep)
%SA_BENCH_SIMSCAPE Simscape 真値（閉ループ、ステップ応答 1.8 s）の逐次の計算時間と実時間比（issue #91）。
%   sa_bench_simscape('reports/issue-91/speed_simscape.json', struct('f',20), 3)
%   モデルの構築・初回コンパイルは除き、2 回目以降の sim の壁時計時間を測る（G-bench v1 と同じシナリオ: P0、軸 1 に 2° のステップ、wn 100、ζ 2、ペイロード 0 kg）。
%   elastic = [] で剛体の真値。-singleCompThread で起動して 1 スレッドにそろえる。
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
if nargin < 2, elastic = struct('f',20); end
if nargin < 3, nRep = 3; end
nAx = 6;  c = sa_config();  dt = 1/c.fsSim;  sp = saParams(nAx);
P0 = [0.0 -0.5 0.5 0.4 0.8 -0.4];  tEnd = 1.8;  tVec = (0:dt:tEnd)';  nT = numel(tVec);
qr = repmat(P0, nT, 1);  qr(tVec >= 0.6, 1) = P0(1) + deg2rad(2);
REF = [tVec, qr, zeros(nT, nAx), saFeedforward(tVec, qr, zeros(nT,nAx), zeros(nT,nAx), struct('mass',0))];
workDir = tempname;  mkdir(workDir);  oldDir = cd(workDir);  cl = onCleanup(@() cd(oldDir)); %#ok<NASGU>
bo = struct('modelName','sa_bench6','slxPath',fullfile(workDir,'sa_bench6.slx'),'payload',struct('mass',0));
if ~isempty(elastic), bo.elastic = elastic; end
tb = tic;  mdl = buildSaClosedLoop(nAx, bo);  tBuild = toc(tb);
wn = 100;  z = 2;
set_param([mdl '/Kp'],'Gain',mat2str(sp.Mmean*wn^2,12));  set_param([mdl '/Kd'],'Gain',mat2str(2*z*sp.Mmean*wn,12));
tc = tic;  simSaClosed(mdl, tVec, REF, [], P0');  tFirst = toc(tc);
t = zeros(1, nRep);
for k = 1:nRep
    ts = tic;  simSaClosed(mdl, tVec, REF, [], P0');  t(k) = toc(ts);
end
close_system(mdl, 0);
r = struct('build_s', tBuild, 'first_sim_s', tFirst, 'sim_s', t, 'sim_time_s', tEnd, ...
    'ms_per_step', 1e3*min(t)/(tEnd*1000), 'realtime_ratio', min(t)/tEnd, 'elastic', ~isempty(elastic));
fprintf('Simscape（弾性=%d）: 構築 %.1f s、初回 %.1f s、2 回目以降 %s s → 実時間比 %.2f（%.3f ms/step）\n', r.elastic, tBuild, tFirst, mat2str(t,3), r.realtime_ratio, r.ms_per_step);
if nargin >= 1 && ~isempty(outFile)
    if ~startsWith(outFile, filesep), outFile = fullfile(repo, outFile); end
    fid = fopen(outFile, 'w');  fprintf(fid, '%s', jsonencode(r, 'PrettyPrint', true));  fclose(fid);
end
end
