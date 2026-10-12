function o = phase2_run(mdl, S, outputMode)
%PHASE2_RUN シナリオ S をモデル mdl で実行する。outputMode: 'default'（ソルバ刻みの出力）または 'grid'（SpecifiedOutputTimes）。
w = warning('off','all');  restore = onCleanup(@() warning(w));
setSaInitialState(mdl, S.q0, zeros(S.nAx,1));
assignin('base','SA_tau_in',[S.t S.tau]);
args = {'StopTime',num2str(S.t(end),'%.12g'),'LoadExternalInput','on','ExternalInput','SA_tau_in', ...
        'SaveOutput','on','SaveFormat','Dataset'};
if strcmp(outputMode,'grid')
    assignin('base','SA_out_times',S.t);
    args = [args {'OutputOption','SpecifiedOutputTimes','OutputTimes','SA_out_times'}];
end
tic; so = sim(mdl, args{:}); o.wall = toc;
o.t = so.yout.getElement(1).Values.Time;  o.q = so.yout.getElement(1).Values.Data;  o.dq = so.yout.getElement(2).Values.Data;
end
