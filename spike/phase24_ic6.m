% 6 軸シナリオの初期条件警告と、初期角速度が正しく与えられているかの確認
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));  clear functions;
tmp = tempname;  mkdir(tmp);
for nAx = [3 6]
    nm = sprintf('sa_excite%d_ic',nAx);
    m = buildSaExcitation(nAx, struct('modelName',nm,'slxPath',fullfile(tmp,[nm '.slx'])));
    S = saScenarioSet(nAx,'small');  s = S(strcmp({S.name},'bln_normal_001'));
    lastwarn('');  raw = runSaScenario(s, struct('excite',m,'closed',m));
    [w,id] = lastwarn;
    fprintf('nAx=%d  warning="%s" (%s)\n', nAx, w, id);
    fprintf('  dq0 - dq(1) max abs = %.3g rad/s (dq0 max %.3g)\n', max(abs(raw.dq(1,:) - s.dq0)), max(abs(s.dq0)));
    fprintf('  q0 - q(1) max abs = %.3g rad\n', max(abs(raw.q(1,:) - s.q0)));
    close_system(m,0);
end
rmdir(tmp,'s');
