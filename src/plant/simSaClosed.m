function out = simSaClosed(mdl, tVec, REF, tauDist, q0, dq0)
%SIMSACLOSED 閉ループモデルを参照軌道で実行する。
%   out = SIMSACLOSED(mdl, tVec, REF, tauDist, q0, dq0)
%     REF [n x (1+3·nAx)] = saPtpReference の出力、tauDist [n x nAx] または []（外乱なし）、q0, dq0 [nAx x 1]
%   out: .t, .q, .dq, .tau（飽和後の実印加トルク）、.qRef, .dqRef（t 上に補間した参照）
nAx = numel(q0);  tVec = tVec(:);
if nargin < 4 || isempty(tauDist), tauDist = zeros(numel(tVec), nAx); end
if nargin < 6 || isempty(dq0), dq0 = zeros(nAx,1); end
assert(size(REF,1) == numel(tVec) && size(tauDist,1) == numel(tVec), 'simSaClosed:size', 'tVec・REF・tauDist の行数が一致しません');
if any(dq0 ~= 0)
    wst = warning('off','physmod:simscape:engine:core:dae_errors:IcDiagnostics');
    restoreWarn = onCleanup(@() warning(wst));
end
setSaInitialState(mdl, q0, dq0);
assignin('base','SA_ref_in',[tVec REF(:,2:end) tauDist]);
assignin('base','SA_ref_t',tVec);
o = sim(mdl, 'StopTime', sprintf('%.17g',tVec(end)), 'LoadExternalInput','on', ...
    'ExternalInput','SA_ref_in', 'SaveOutput','on', 'SaveFormat','Dataset', ...
    'OutputOption','SpecifiedOutputTimes', 'OutputTimes','SA_ref_t');
out.t  = o.yout.getElement(1).Values.Time;
out.q  = o.yout.getElement(1).Values.Data;
out.dq = o.yout.getElement(2).Values.Data;
out.tau = o.yout.getElement(3).Values.Data;
out.qRef = interp1(tVec, REF(:,2:1+nAx), out.t);
out.dqRef = interp1(tVec, REF(:,2+nAx:1+2*nAx), out.t);
end
