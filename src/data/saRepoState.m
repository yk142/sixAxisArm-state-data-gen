function r = saRepoState()
%SAREPOSTATE 本リポジトリの来歴（HEAD のコミット、未コミット差分の有無とそのハッシュ）を返す。
%   r.commit      HEAD の SHA（取得できなければ ''）
%   r.dirty       追跡ファイルに未コミットの差分があるか（data/ などの無視ファイルは含まない。未追跡の新規ファイルは含む）
%   r.diffHash    dirty のとき、`git diff HEAD` と未追跡ファイル名一覧の SHA-256 先頭 16 桁（dirty でなければ ''）
root = sa_config().rootDir;
r = struct('commit','','dirty',false,'diffHash','');
[st, out] = system(sprintf('git --no-pager -C "%s" rev-parse HEAD', root));
if st ~= 0, return; end
r.commit = strtrim(out);
[st, out] = system(sprintf('git --no-pager -C "%s" status --porcelain', root));
if st ~= 0, return; end
r.dirty = ~isempty(strtrim(out));
if r.dirty
    [~, d] = system(sprintf('git --no-pager -C "%s" diff HEAD', root));
    [~, u] = system(sprintf('git --no-pager -C "%s" ls-files --others --exclude-standard', root));
    try
        md = java.security.MessageDigest.getInstance('SHA-256');
        md.update(uint8([d u]));
        h = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'))', 1, []));
        r.diffHash = h(1:16);
    catch
    end
end
end
