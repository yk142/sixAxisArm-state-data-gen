function names = saReproSample(S, mode)
%SAREPROSAMPLE 再現性確認に使うシナリオ名を選ぶ。
%   names = SAREPROSAMPLE(S, 'sample')  各パターンの先頭 1 本＋件数が 8 本以上のパターンは末尾 1 本も（3 軸 full で 31 本）
%   names = SAREPROSAMPLE(S, 'all')     全件
if strcmp(mode,'all'), names = {S.name}; return; end
pats = unique({S.pattern}, 'stable');  names = {};
for i = 1:numel(pats)
    idx = find(strcmp({S.pattern}, pats{i}));
    pick = idx(1);
    if numel(idx) >= 8, pick = [pick idx(end)]; end
    names = [names, {S(pick).name}]; %#ok<AGROW>
end
end
