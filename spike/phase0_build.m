function mdl = phase0_build(nAx, outDir, opts)
%PHASE0_BUILD 先頭 nAx 軸を可動、残りを 0 rad で fixed にした Simscape プラントを作る（nAx=3 or 6）。
%   親資産の writeC8URDF を後処理して smimport。可動軸は joint1..nAx のまま
%   addC8Actuation/Armature/Friction（nJoints=nAx）を適用する。
if nargin < 3, opts = struct(); end
if ~isfield(opts,'armature'), opts.armature = true; end
if ~isfield(opts,'friction'), opts.friction = true; end
if ~isfield(opts,'suffix'), opts.suffix = ''; end
c = sa_config();
p = c8_params(c.meshDir);
name = sprintf('sa_plant%d%s', nAx, opts.suffix);
if ~isfolder(outDir), mkdir(outDir); end
urdf6 = fullfile(outDir,[name '_6ax.urdf']);
urdf1 = fullfile(outDir,[name '.urdf']);
writeC8URDF(p, urdf6);
txt = fileread(urdf6);
for i = nAx+1:6
    blk = regexp(txt, sprintf('<joint name="joint%d" type="revolute">.*?</joint>', i), 'match', 'once');
    assert(~isempty(blk), 'joint%d が見つかりません', i);
    nb = strrep(blk, 'type="revolute"', 'type="fixed"');
    nb = regexprep(nb, '\s*<axis[^>]*/>', '');
    nb = regexprep(nb, '\s*<limit[^>]*/>', '');
    txt = strrep(txt, blk, nb);
end
fid = fopen(urdf1,'w'); fwrite(fid, txt); fclose(fid);
if bdIsLoaded(name), close_system(name,0); end
smimport(urdf1, 'ModelName', name);
mc = find_system(name,'MaskType','Mechanism Configuration');
set_param(mc{1}, 'GravityVector', sprintf('[%.9g %.9g %.9g]', p.gravity));
set_param(name,'SolverType','Fixed-step','Solver','ode14x', ...
    'FixedStep',num2str(1/c.fsSim,'%.12g'),'StopTime','0.5','SimscapeLogType','none');
try, set_param(name,'SimMechanicsOpenEditorOnUpdate','off'); catch, end
pn = p; pn.nJoints = nAx;
addC8Actuation(name, pn);
if opts.armature, addC8Armature(name, pn); end
if opts.friction, addC8Friction(name, pn); end
save_system(name, fullfile(outDir,[name '.slx']));
mdl = name;
fprintf('phase0_build: %s revolute=%d\n', name, numel(find_system(name,'MaskType','Revolute Joint')));
end
