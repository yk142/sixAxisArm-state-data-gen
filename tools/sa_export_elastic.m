function sa_export_elastic(outFile, e)
%SA_EXPORT_ELASTIC 関節弾性のパラメータ（k, c, Jl, Jm, f）を JSON に書く（2 慣性の解析モデル python/sa_nsm/twomass.py 用、issue #77）。
%   sa_export_elastic('data/export/elastic20.json')   % 既定（f = 20 Hz, ζ = 0.05, thetaMax 0.02）。#71 の弾性真値と同じ値
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
if nargin < 1, outFile = fullfile(repo,'data/export/elastic20.json'); end
if nargin < 2, e = struct(); end
el = saElasticParams(6, e);
o = struct('k',el.k,'c',el.c,'Jl',el.Jl,'Jm',el.Jm,'f',el.f);
fid = fopen(outFile,'w'); fwrite(fid, jsonencode(o)); fclose(fid);
end
