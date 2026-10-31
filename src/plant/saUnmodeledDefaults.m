function um = saUnmodeledDefaults(um)
%SAUNMODELEDDEFAULTS 解析モデルの式にない効果（issue #97）の既定値を埋める。
%   stiffA 0.3・stiffThetaRef 0.005 rad（非線形剛性）、stribeck 0.3（Fc 比）・vsMul 2（vs = vsMul·ε）
if nargin < 1 || isempty(um), um = struct(); end
d = struct('stiffA',0.3,'stiffThetaRef',0.005,'stribeck',0.3,'vsMul',2);
for f = fieldnames(d)', if ~isfield(um,f{1}) || isempty(um.(f{1})), um.(f{1}) = d.(f{1}); end, end
end
