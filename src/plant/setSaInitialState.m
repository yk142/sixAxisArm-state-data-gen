function setSaInitialState(mdl, q0, dq0)
%SETSAINITIALSTATE プラントの初期角度・初期角速度を設定する。
%   SETSAINITIALSTATE(mdl, q0, dq0)   q0, dq0: [nAx x 1]（dq0 省略時は 0）
q0 = q0(:);
if nargin < 3, dq0 = zeros(size(q0)); end
dq0 = dq0(:);
for i = 1:numel(q0)
    set_param(sprintf('%s/joint%d',mdl,i), ...
        'PositionTargetSpecify','on', 'PositionTargetValue',sprintf('%.12g',q0(i)), 'PositionTargetValueUnits','rad', ...
        'VelocityTargetSpecify','on', 'VelocityTargetValue',sprintf('%.12g',dq0(i)), 'VelocityTargetValueUnits','rad/s');
    sn = sprintf('%s/motorSns_%d',mdl,i);          % 関節弾性つき: モータ側センサの角度は q0 から始める（issue #71）
    if getSimulinkBlockHandle(sn) > 0, set_param(sn,'offset',sprintf('%.12g',q0(i)),'offset_unit','rad'); end
end
end
