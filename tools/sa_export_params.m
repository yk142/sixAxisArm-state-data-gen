function sa_export_params(outFile, nTest, payload)
%SA_EXPORT_PARAMS 解析モデルのパラメータと、検証用の MATLAB 側の前進動力学の出力を JSON に書く（Python 移植の検証用、issue #33）。
%   sa_export_params('data/export/c8_params.json', 200)
%   sa_export_params('data/export/c8_params_pl3.json', 200, struct('mass',3))   % 手先ペイロード付き（issue #61。saApplyPayload）
%   p        : c8_params（運動学・リンク慣性・armature・摩擦・ε・可動範囲・トルク容量・重力）
%   test     : nAx = 3 と 6 の乱数状態（q, dq, tau）と、saFwdDyn の加速度（ddq）、g(q) と M(q) の対角
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
if nargin < 2, nTest = 200; end
p = c8_params(sa_config().meshDir);
if nargin >= 3 && ~isempty(payload), p = saApplyPayload(p, payload); end
o = struct();
o.gravity = p.gravity;  o.baseRpy = p.baseOrigin.rpy;
for i = 1:6
    L = p.links(i+1);
    o.joint(i) = struct('xyz', p.joint(i).xyz, 'rpy', p.joint(i).rpy);
    o.link(i) = struct('mass', L.mass, 'com', L.com(:)', 'inertiaC', L.inertiaC);
end
for f = {'armature','frictionFc','frictionBv','frictionEps','qMin','qMax','qdMax','tauPeak','tauRated'}
    o.(f{1}) = p.(f{1});
end
rs = RandStream('twister','Seed',20261014);
mid = (p.qMax + p.qMin)/2;  half = (p.qMax - p.qMin)/2;
o.test = struct('nAx',{},'q',{},'dq',{},'tau',{},'ddq',{});
for nAx = [3 6]
    q = mid(1:nAx) + 0.6*half(1:nAx).*(2*rand(rs,nTest,nAx)-1);
    dq = p.qdMax(1:nAx).*0.5.*(2*rand(rs,nTest,nAx)-1);
    tau = p.tauRated(1:nAx).*(2*rand(rs,nTest,nAx)-1);
    ddq = zeros(nTest,nAx);
    for k = 1:nTest, ddq(k,:) = saFwdDyn(p, q(k,:)', dq(k,:)', tau(k,:)')'; end
    o.test(end+1) = struct('nAx',nAx,'q',q,'dq',dq,'tau',tau,'ddq',ddq);
end
if ~isfolder(fileparts(outFile)), mkdir(fileparts(outFile)); end
fid = fopen(outFile, 'w');  fwrite(fid, jsonencode(o, 'PrettyPrint', false), 'char');  fclose(fid);
fprintf('書き出し: %s\n', outFile);
end
