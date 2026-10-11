function out = simSa(mdl, tVec, tau, q0, dq0)
%SIMSA プラントを指定トルク列で実行し、状態の時系列を返す。
%   out = SIMSA(mdl, tVec, tau, q0, dq0)
%     tVec [N x 1] 時刻列（先頭 0）、tau [N x nAx] または [1 x nAx]（一定トルク）、q0, dq0 [nAx x 1]
%   out: .t [M x 1], .q [M x nAx], .dq [M x nAx]。トルクは tVec 上のゼロ次保持。
tVec = tVec(:);  q0 = q0(:);  nAx = numel(q0);
if nargin < 5, dq0 = zeros(nAx,1); end
if size(tau,1) == 1, tau = repmat(tau, numel(tVec), 1); end
assert(size(tau,1) == numel(tVec) && size(tau,2) == nAx, 'simSa:size', 'tau は [numel(tVec) x nAx]');
if any(dq0 ~= 0)   % 初期速度があると armature の回転子初期条件の警告が出る（関節速度は満たされる）
    wst = warning('off','physmod:simscape:engine:core:dae_errors:IcDiagnostics');
    restoreWarn = onCleanup(@() warning(wst));
end
setSaInitialState(mdl, q0, dq0);
tt = reshape([tVec(1:end-1) tVec(2:end)-eps(tVec(2:end))]', [], 1);
uu = kron(tau(1:end-1,:), [1; 1]);
tt = [tt; tVec(end)];  uu = [uu; tau(end,:)];
assignin('base','SA_tau_in',[tt uu]);
o = sim(mdl, 'StopTime', num2str(tVec(end),'%.12g'), 'LoadExternalInput','on', ...
    'ExternalInput','SA_tau_in', 'SaveOutput','on', 'SaveFormat','Dataset');
out.t  = o.yout.getElement(1).Values.Time;
out.q  = o.yout.getElement(1).Values.Data;
out.dq = o.yout.getElement(2).Values.Data;
end
