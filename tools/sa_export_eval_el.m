function sa_export_eval_el(outFile)
%SA_EXPORT_EVAL_EL 弾性真値のキャッシュ（data/cache/fullel6_*.mat）から、閉ループ（ptp / hold）と励振の評価用データを HDF5 に書く（issue #79）。Simscape は再実行しない。
%   sa_export_eval_el('data/export/eval6_el_scen.h5')
%   閉ループ（ptp / hold）: ref [N x 19]、tau_d（saClosedInputs。closed_eval6.h5 と同じ形式）、truth_q / dq / qm / dqm（8 kHz）
%   励振: truth_tau（実印加トルク。ソフトバリア込み）も書く。ref / tau_d は無い
%   属性: type, pattern, split, contact（リミット拘束 = 可動範囲の端から 0.1° 以内または外側の点の割合）
repo = fileparts(fileparts(mfilename('fullpath')));
if ~startsWith(outFile, filesep), outFile = fullfile(pwd, outFile); end
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
S = [saScenarioSetPayload(6,'full',0,[]), saScenarioSetPulse(6,120,12,[0 1 2 4 6 8])];
cdir = fullfile(repo,'data/cache');
has = arrayfun(@(s) isfile(fullfile(cdir, sprintf('fullel6_%s.mat', s.name))), S);
S = S(has);
pick = false(size(S));
nPer = struct('ptp',2,'hold',1,'excitation',1);
pats = unique({S.pattern}, 'stable');
for i = 1:numel(pats)
    idx = find(strcmp({S.pattern}, pats{i}) & ~contains({S.name}, '_r1'));
    ty = S(idx(1)).type;
    if strcmp(ty,'excitation') && (startsWith(pats{i},'contact') || strcmp(pats{i},'nearlimit')), continue; end   % リミット接触の専用シナリオは対象外
    if strcmp(pats{i},'nearlimit'), continue; end
    k = nPer.(ty);
    if startsWith(pats{i},'ptp'), k = 1; end                % ptp は速度違い 4 種を 1 本ずつ
    if strcmp(ty,'excitation') && strcmp(pats{i},'pulse')           % パルスは軸ごとに 1 本
        [~, ia] = unique([S(idx).axis], 'stable');  pick(idx(ia)) = true;  continue
    end
    pick(idx(1:min(k,numel(idx)))) = true;
end
S = S(pick);
if exist(outFile,'file'), delete(outFile); end
for i = 1:numel(S)
    s = S(i);  t0 = tic;
    x = load(fullfile(cdir, sprintf('fullel6_%s.mat', s.name)));  e = x.e;
    g = ['/' s.name];
    w = @(k, v) wr(outFile, [g '/' k], v);
    if ismember(s.type, {'hold','ptp'})
        [REF, tauD] = saClosedInputs(s);
        assert(max(abs(e.qref - REF(:,2:7)), [], 'all') < 1e-9, '参照の再構成がキャッシュと一致しません: %s', s.name);
        w('ref', REF);  w('tau_d', tauD);
    else
        w('truth_tau', e.tau);
    end
    w('truth_q', e.q);  w('truth_dq', e.dq);  w('truth_qm', e.qm);  w('truth_dqm', e.dqm);
    h5writeatt(outFile, g, 'type', s.type);  h5writeatt(outFile, g, 'pattern', s.pattern);  h5writeatt(outFile, g, 'split', s.split);
    h5writeatt(outFile, g, 'contact', double(mean(e.contact8k(:))));
    h5writeatt(outFile, g, 'payload', double(e.payload));
    fprintf('%2d/%d %s (%s/%s) %.1f s\n', i, numel(S), s.name, s.type, s.pattern, toc(t0));
end
sp = saParams(6);
wr(outFile, '/controller/Mmean', sp.Mmean(:)');  wr(outFile, '/controller/tauPeak', sp.tauPeak(:)');
wr(outFile, '/controller/wn', 60);  wr(outFile, '/controller/zeta', 1);
end

function wr(f, name, v)
h5create(f, name, size(v), 'Datatype', 'double');
h5write(f, name, double(v));
end
