function el = saUnmodeledElastic(el, um)
%SAUNMODELEDELASTIC 非線形剛性（issue #97）の設定を el（saElasticParams の出力）に載せる。addSaElasticArmature が読んで、線形ばねの代わりに
%   Simscape 言語の部品 saSsc.nl_spring（src/plant/ssc/+saSsc/nl_spring.ssc）を置く。部品ライブラリ saSsc_lib.slx が無ければ ssc_build で作る。
um = saUnmodeledDefaults(um);
el.stiffA = um.stiffA;  el.stiffThetaRef = um.stiffThetaRef;
if um.stiffA ~= 0, ensureSscLib(); end
end

function ensureSscLib()
dd = fullfile(fileparts(mfilename('fullpath')), 'ssc');
addpath(dd);
if ~isfile(fullfile(dd, 'saSsc_lib.slx'))
    old = cd(dd);  c = onCleanup(@() cd(old));
    ssc_build('saSsc');
end
load_system('saSsc_lib');
end
