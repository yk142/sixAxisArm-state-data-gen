function out = sa_eval_report(dsFile, outDir, nAx)
%SA_EVAL_REPORT カバレッジ・解析モデル整合の評価。指標を mat と PNG（outDir）に出力する。
%   sa_eval_report('data/sa_dataset_full3.mat', 'reports/issue-23', 3)
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
if ~isfolder(outDir), mkdir(outDir); end
ds = load(dsFile);  sp = ds.manifest.params;  sc = ds.scen;
r = saCoverage(ds);
T = saAnalyticResidual(ds, 100);
col = struct('train',[0.2 0.4 0.8],'val',[0.1 0.7 0.3],'test',[0.9 0.6 0.1],'benchmark',[0.85 0.1 0.1]);
mid = (sp.qMax + sp.qMin)/2;  half = (sp.qMax - sp.qMin)/2;
% ---- 1. 軸別位相平面（train を背景、benchmark を重ねる）----
f = figure('Visible','off','Position',[0 0 1500 450]);
for a = 1:nAx
    subplot(1,nAx,a);  hold on;
    X = r.X.train;  ii = 1:20:size(X,1);
    h1 = plot(X(ii,a), X(ii,nAx+a), '.', 'Color', col.train, 'MarkerSize', 1);
    B = r.X.benchmark;  h2 = plot(B(1:5:end,a), B(1:5:end,nAx+a), '.', 'Color', col.benchmark, 'MarkerSize', 3);
    xlabel(sprintf('q_%d（正規化）',a)); ylabel(sprintf('dq_%d / qdMax',a)); grid on; xlim([-1 1]); ylim([-1 1]);
    title(sprintf('J%d: train が占めるセル %.0f%%、benchmark の被覆 %.1f%%', a, 100*r.axisFill(a), 100*r.axisPlane(a,3)));
end
legend([h1 h2], {'train（1/20 に間引き）','benchmark（1/5 に間引き）'}, 'Location','southoutside');
exportgraphics(f, fullfile(outDir,'phase_plane.png'), 'Resolution', 110);  close(f);
% ---- 2. トルク分布（train と benchmark、軸別）----
f = figure('Visible','off','Position',[0 0 1500 400]);
tr = vertcat(sc(strcmp({sc.split},'train')).tau);  bm = vertcat(sc(strcmp({sc.split},'benchmark')).tau);
for a = 1:nAx
    subplot(1,nAx,a);  e = linspace(-sp.tauPeak(a), sp.tauPeak(a), 61);
    histogram(tr(:,a), e, 'Normalization','probability', 'FaceColor', col.train); hold on;
    histogram(bm(:,a), e, 'Normalization','probability', 'FaceColor', col.benchmark, 'FaceAlpha', 0.5);
    set(gca,'YScale','log'); xlabel(sprintf('\\tau_%d [N·m]',a)); ylabel('割合'); grid on; title(sprintf('J%d', a));
end
legend({'train','benchmark'});
exportgraphics(f, fullfile(outDir,'torque_hist.png'), 'Resolution', 110);  close(f);
% ---- 3. トルクの周波数成分（train の全シナリオの平均パワー、1 kHz、窓 1024 点）----
Nw = 1024;  fs = ds.manifest.fsData;  P = zeros(Nw/2+1, nAx);  cnt = 0;
for k = find(strcmp({sc.split},'train'))
    t = sc(k).tau;  K = size(t,1);
    for s0 = 1:Nw/2:K-Nw+1
        seg = t(s0:s0+Nw-1,:) - mean(t(s0:s0+Nw-1,:),1);  w = 0.5 - 0.5*cos(2*pi*(0:Nw-1)'/Nw);
        Y = abs(fft(seg.*w, [], 1)).^2/sum(w.^2)/fs;  P = P + Y(1:Nw/2+1,:);  cnt = cnt + 1;
    end
end
P = P/cnt;  fr = (0:Nw/2)'*fs/Nw;
f = figure('Visible','off','Position',[0 0 700 450]);
loglog(fr(2:end), P(2:end,:)); grid on; xlabel('周波数 [Hz]'); ylabel('トルクのパワースペクトル密度 [(N·m)^2/Hz]');
legend(arrayfun(@(a) sprintf('J%d',a), 1:nAx, 'UniformOutput', false)); title('train のトルクのスペクトル（平均）');
exportgraphics(f, fullfile(outDir,'torque_spectrum.png'), 'Resolution', 110);  close(f);
% ---- 4. 6 次元ボクセル被覆 ----
f = figure('Visible','off','Position',[0 0 700 420]);
g = r.grid(arrayfun(@(x) x.minCount == 1, r.grid));
bar(categorical(string([g.nBins]) + " 分割"), 100*[[g.val]' [g.test]' [g.benchmark]']);
ylabel('train に同じセルの点がある割合 [%]'); legend({'val','test','benchmark（PTP）'}, 'Location','northeast'); grid on;
title(sprintf('正規化 %d 次元（q, dq）の格子被覆（train %d 点）', 2*nAx, r.n.train));
exportgraphics(f, fullfile(outDir,'voxel_coverage.png'), 'Resolution', 110);  close(f);
% ---- 5. 解析モデルとの整合（パターン別、軸別の 残差 RMS / 加速度 RMS）----
pats = unique(T.pattern, 'stable');  R = zeros(numel(pats), nAx);
for i = 1:numel(pats)
    m = strcmp(T.pattern, pats{i});
    R(i,:) = sqrt(mean(T.rmsRes(m,:).^2,1)) ./ sqrt(mean(T.rmsAcc(m,:).^2,1));
end
f = figure('Visible','off','Position',[0 0 1100 420]);
bar(categorical(pats, pats), 100*R); set(gca,'YScale','log'); ylabel('残差 RMS / 加速度 RMS [%]'); grid on;
set(gca,'TickLabelInterpreter','none'); legend(arrayfun(@(a) sprintf('J%d',a), 1:nAx, 'UniformOutput', false)); title('解析モデル（RNEA＋摩擦）との 1 ステップ整合（パターン別）');
exportgraphics(f, fullfile(outDir,'analytic_residual.png'), 'Resolution', 110);  close(f);
out = struct('cov', rmfield(r,'X'), 'resid', T, 'residPattern', array2table(R, 'RowNames', pats), 'spectrum', struct('f',fr,'P',P));
save(fullfile(outDir,'eval_metrics.mat'), '-struct', 'out');
disp(out.residPattern); disp(r.dqTrain999); disp(r.dqTrainMax); disp(r.dqOver.benchmark);
end
