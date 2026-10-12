function r = saCompareCache(dirA, dirB, names, preset, nAx)
%SACOMPARECACHE 2 つのキャッシュフォルダの同名シナリオを比較する（ビット一致の確認）。
%   r(i): name, identical（全フィールドが isequal）, maxAbsDiff（q,dq,tau の最大差）, firstDiffStep（最初に違う 8 kHz 点、一致なら 0）
r = struct('name',{},'identical',{},'maxAbsDiff',{},'firstDiffStep',{});
for i = 1:numel(names)
    fn = sprintf('%s%d_%s.mat', preset, nAx, names{i});
    a = load(fullfile(dirA, fn), 'e');  b = load(fullfile(dirB, fn), 'e');
    f = {'q','dq','tau','qref','dqref','q0','dq0','contact8k'};
    same = all(cellfun(@(k) isequal(a.e.(k), b.e.(k)), f));
    d = 0;  first = 0;
    if ~same
        for k = {'q','dq','tau'}
            x = a.e.(k{1});  y = b.e.(k{1});
            if ~isequal(size(x), size(y)), d = Inf; first = 1; continue; end
            dd = abs(x - y);  d = max(d, max(dd, [], 'all'));
            j = find(any(dd > 0, 2), 1);  if ~isempty(j) && (first == 0 || j < first), first = j; end
        end
    end
    r(i) = struct('name',names{i},'identical',same,'maxAbsDiff',d,'firstDiffStep',first);
end
end
