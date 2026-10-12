function [qSoftMin, qSoftMax, dqSoft] = setSaBarrier(mdl, sp, qSoftFrac, dqSoftFrac)
%SETSABARRIER 励振モデルのソフトバリアの安全帯を設定する（軸ごと）。
%   [qSoftMin, qSoftMax, dqSoft] = SETSABARRIER(mdl, sp, qSoftFrac, dqSoftFrac)
%     sp          saParams の出力
%     qSoftFrac   可動域に対する安全帯の比率（可動域の中心から ±frac·半幅。1.0 でリミット位置、
%                 1.0 超でリミットの外側）。スカラまたは [1 x nAx]
%     dqSoftFrac  最大角速度に対する速度安全帯の比率。スカラまたは [1 x nAx]
%   バリア位置は引数。Phase 3 の接触シナリオ用に、リミットの外側（frac > 1）へ移せる。
mid  = (sp.qMax + sp.qMin)/2;  half = (sp.qMax - sp.qMin)/2 .* qSoftFrac;
qSoftMin = mid - half;  qSoftMax = mid + half;  dqSoft = sp.qdMax .* dqSoftFrac;
set_param([mdl '/qDead'],  'LowerValue',mat2str(qSoftMin,12), 'UpperValue',mat2str(qSoftMax,12));
set_param([mdl '/dqDead'], 'LowerValue',mat2str(-dqSoft,12),  'UpperValue',mat2str(dqSoft,12));
end
