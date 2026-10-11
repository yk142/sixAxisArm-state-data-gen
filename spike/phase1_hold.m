function r = phase1_hold(mdl, q0, tg, T)
%PHASE1_HOLD 重力補償トルクで T 秒保持（格子入力）。ドリフト[deg]・最大|dq|・実行時間を文字列で返す。失敗時は ERR。
if nargin < 4, T = 1; end
fs = 8000;  w = warning('off','all');  restore = onCleanup(@() warning(w));
t = (0:1/fs:T)';  tgrid = repmat(tg(:)', numel(t), 1);
try
    tic;  o = simSa(mdl, t, tgrid, q0, [], true);  wt = toc;
    r = sprintf('drift=%.3g deg maxdq=%.3g (%.0f s)', max(abs(rad2deg(o.q - q0(:)')),[],'all'), max(abs(o.dq),[],'all'), wt);
catch ee
    r = ['ERR ' strtok(ee.message, newline)];
end
end
