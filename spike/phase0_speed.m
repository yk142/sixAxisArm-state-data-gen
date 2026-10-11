function T = phase0_speed(nAx, qh)
%PHASE0_SPEED 実行速度の切り分け（issue #3）。同一条件の保持 2 s を 2 回走らせ、2 回目の時間を採る。
c = sa_config();  p = c8_params(c.meshDir);  fs = c.fsSim;  qh = qh(:);
dd = fullfile(c.dataDir,'models');
cases = {'base','',struct(); 'noArm','_noarm',struct('armature',false,'suffix','_noarm'); ...
         'noFric','_nofr',struct('friction',false,'suffix','_nofr')};
rows = {};
for k = 1:size(cases,1)
    if k > 1, phase0_build(nAx, dd, cases{k,3}); end
    rows(end+1,:) = runCase(sprintf('sa_plant%d%s',nAx,cases{k,2}), cases{k,1}, nAx, qh, p, fs, dd, {}); %#ok<AGROW>
end
mdl = sprintf('sa_plant%d',nAx);
rows(end+1,:) = runCase(mdl,'newton2',nAx,qh,p,fs,dd,{'NumberNewtonIterations','2'});
rows(end+1,:) = runCase(mdl,'newton3',nAx,qh,p,fs,dd,{'NumberNewtonIterations','3'});
rows(end+1,:) = runCase(mdl,'rapid',nAx,qh,p,fs,dd,{'SimulationMode','rapid-accelerator'});
T = cell2table(rows,'VariableNames',{'case','wall1','wall2','xRT','driftDeg'});
disp(T);
end

function r = runCase(mdl, label, nAx, qh, p, fs, dd, prm)
if ~bdIsLoaded(mdl), load_system(fullfile(dd,[mdl '.slx'])); end
orig = cell(1,numel(prm)/2);
for i = 1:2:numel(prm), orig{(i+1)/2} = get_param(mdl,prm{i}); set_param(mdl,prm{i},prm{i+1}); end
for i = 1:nAx
    set_param(sprintf('%s/joint%d',mdl,i),'PositionTargetSpecify','on', ...
        'PositionTargetValue',sprintf('%.12g',qh(i)),'PositionTargetValueUnits','rad');
end
tg = c8RNEA([qh(1:nAx);zeros(6-nAx,1)],zeros(6,1),zeros(6,1),p);  tg = tg(1:nAx);
Tn = 2;  t = (0:1/fs:Tn)';  assignin('base','SA_in',[t repmat(tg',numel(t),1)]);
w = zeros(1,2);
for k = 1:2
    tic; o = sim(mdl,'StopTime',num2str(Tn),'LoadExternalInput','on','ExternalInput','SA_in', ...
        'SaveOutput','on','SaveFormat','Dataset'); w(k) = toc;
end
q = o.yout.getElement(1).Values.Data;
r = {label, w(1), w(2), w(2)/Tn, max(rad2deg(abs(q-q(1,:))),[],'all')};
for i = 1:2:numel(prm), set_param(mdl,prm{i},orig{(i+1)/2}); end
end
