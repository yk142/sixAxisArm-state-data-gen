classdef test_sa_export < matlab.unittest.TestCase
%TEST_SA_EXPORT HDF5 出力の単体テスト（合成データ。シミュレーションなし。issue #25）。
%   Python（~/.venvs/sa-h5、h5py と scipy）が無い場合は、Python 側の検証を省略する。
    properties
        ds
        tmp
        py = fullfile(getenv('HOME'), '.venvs', 'sa-h5', 'bin', 'python')
    end
    methods (TestClassSetup)
        function setup(tc)
            repo = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
            tc.tmp = tempname;  mkdir(tc.tmp);
            tc.addTeardown(@() rmdir(tc.tmp, 's'));
            tc.ds = makeDs();
        end
    end
    methods (Test)
        function exportAndLayout(tc)
            fn = fullfile(tc.tmp, 'a.h5');
            info = saExportFlat(tc.ds, fn);
            tc.verifyEqual(info.n.train, sum([tc.ds.scen(strcmp({tc.ds.scen.split},'train')).nTransition]));
            tc.verifyEqual(h5info(fn,'/train/X').Dataspace.Size, [9 info.n.train]);       % MATLAB 側は (d, N)。h5py からは (N, d)
            tc.verifyEqual(h5info(fn,'/train/X').Datatype.Class, 'H5T_FLOAT');
            tc.verifyEqual(h5info(fn,'/train/X').Datatype.Size, 4);                        % single
            start = h5read(fn,'/scenarios/start');  n = h5read(fn,'/scenarios/n_transition');  sp = h5read(fn,'/scenarios/split');
            tc.verifyEqual(double(start(strcmp(sp,'train'))), [0; 50; 120]);                 % 長さ 50, 70, 60 の累積
            X = h5read(fn,'/train/X')';  s1 = tc.ds.scen(find(strcmp({tc.ds.scen.split},'train'),1));
            tc.verifyEqual(X(1:s1.nTransition,1:3), single(s1.q(1:end-1,:)));
        end
        function pythonVerifies(tc)
            tc.assumeTrue(isfile(tc.py), 'Python 環境（~/.venvs/sa-h5）が無いため省略');
            fn = fullfile(tc.tmp, 'b.h5');  saExportFlat(tc.ds, fn);
            repo = fileparts(fileparts(mfilename('fullpath')));
            [st, out] = system(sprintf('"%s" -I "%s" "%s"', tc.py, fullfile(repo,'tools','verify_h5.py'), fn));
            tc.verifyEqual(st, 0, out);
            tc.verifyTrue(contains(out, '項目 OK') && ~contains(out, '[NG]'), out);
        end
        function pythonDetectsCorruption(tc)
            tc.assumeTrue(isfile(tc.py), 'Python 環境（~/.venvs/sa-h5）が無いため省略');
            fn = fullfile(tc.tmp, 'c.h5');  saExportFlat(tc.ds, fn);
            h5write(fn, '/train/scenario_id', h5read(fn,'/train/scenario_id') + int32(1));      % シナリオ ID を壊す
            repo = fileparts(fileparts(mfilename('fullpath')));
            [st, out] = system(sprintf('"%s" -I "%s" "%s"', tc.py, fullfile(repo,'tools','verify_h5.py'), fn));
            tc.verifyNotEqual(st, 0);  tc.verifyTrue(contains(out, '[NG]'));
        end
    end
end

function ds = makeDs()
% 3 軸の合成データセット（train 3 本・val 1 本・test 1 本・benchmark 1 本）
rng(3);  nAx = 3;  sp = saParams(3);  dt = 1e-3;
split = {'train','train','train','val','test','benchmark'};  K = [50 70 60 40 45 80];
mid = (sp.qMax+sp.qMin)/2;
for i = 1:6
    q = mid + 0.3*(sp.qMax-sp.qMin)/2.*cumsum(0.01*randn(K(i)+1,nAx), 1);
    dq = [diff(q,1,1)/dt; zeros(1,nAx)];  dq = dq/50;
    if i == 6, qref = q(1:K(i),:); else, qref = []; end
    sc(i) = struct('name',sprintf('s%d',i),'pattern','p','type','excitation','phase',0,'split',split{i},'seed',i, ...
        'q',q,'dq',dq,'tau',sp.tauPeak.*(2*rand(K(i),nAx)-1),'tauInst',zeros(K(i),nAx),'qref',qref,'dqref',[], ...
        'atLimit',false(K(i),1),'contact8k',0,'nTransition',K(i)); %#ok<AGROW>
end
isTr = strcmp(split,'train');
X = cell2mat(arrayfun(@(s) [s.q(1:end-1,:), s.dq(1:end-1,:), s.tau], sc(isTr)', 'UniformOutput', false));
Y = cell2mat(arrayfun(@(s) [diff(s.q,1,1), diff(s.dq,1,1)], sc(isTr)', 'UniformOutput', false));
stats = struct('xMean',mean(X,1),'xStd',std(X,0,1),'yMean',mean(Y,1),'yStd',std(Y,0,1),'nTrain',size(X,1));
scale = struct('DQ_MAX',sp.qdMax,'TAU_MAX',sp.tauPeak,'D_Q_MAX',sp.qdMax*dt,'D_DQ_MAX',sp.tauPeak./sp.Mmin*dt);
m = struct('nAx',3,'params',sp,'preset','synthetic','created','now','fsData',1000,'fsRaw',8000,'scenarioHash','abc', ...
    'parentCommit','p','repoCommit','r','matlab','m','genClean',true,'genRepo',{{'c|0|'}});
ds = struct('scen',sc(:),'stats',stats,'scale',scale,'manifest',m);
end
