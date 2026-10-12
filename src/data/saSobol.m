function X = saSobol(n, d, seed)
%SASOBOL Sobol 低食い違い列（最大 12 次元）を自前実装し、[0,1)^d の n 点を返す。
%   X = SASOBOL(n, d)         標準の Sobol 列（先頭点は 0）
%   X = SASOBOL(n, d, seed)   seed を与えると、次元ごとのランダム XOR シフト（digital shift）をかけて
%                             再現可能にばらつかせる（低食い違い性は保たれる）。
%   Statistics and Machine Learning Toolbox の sobolset は使用禁止のため自前で実装する。
%   方向数の初期値は Joe–Kuo の表（次元 2〜12: s, a, m）。点生成は Gray code 法。
%   3 軸は 6 次元（q0 3 + dq0 3）、6 軸は 12 次元（q0 6 + dq0 6）に使う。
%   注意: 方向数の値は Joe–Kuo の表からの転記で、各次元の本原多項式は検証済みだが、
%   公式表との一致は未照合。均等性は test_sa_scenarioset で検査する。
if nargin < 3, seed = []; end
assert(d >= 1 && d <= 12, 'saSobol:dim', '対応次元は 1〜12 です');
B = 32;
S  = [0 1 2 3 3 4 4 5 5 5 5 5];
A  = [0 0 1 1 2 1 4 2 4 7 11 13];
M0 = {[], 1, [1 3], [1 3 1], [1 1 1], [1 1 3 3], [1 3 5 13], [1 1 5 5 17], [1 1 5 5 5], [1 1 7 11 19], [1 1 5 1 1], [1 3 1 1 1]};
V = zeros(d, B, 'uint32');
for k = 1:B, V(1,k) = bitshift(uint32(1), B-k); end
for j = 2:d
    s = S(j);  a = A(j);  m = M0{j};
    for k = 1:s, V(j,k) = bitshift(uint32(m(k)), B-k); end
    for k = s+1:B
        v = bitxor(V(j,k-s), bitshift(V(j,k-s), -s));
        for l = 1:s-1
            if bitget(a, s-l), v = bitxor(v, V(j,k-l)); end
        end
        V(j,k) = v;
    end
end
shift = zeros(1, d, 'uint32');
if ~isempty(seed)
    rs = RandStream('twister','Seed',seed);
    shift = uint32(floor(rand(rs,1,d) * 2^32));
end
x = zeros(1, d, 'uint32');
X = zeros(n, d);
X(1,:) = double(bitxor(x, shift)) / 2^32;
for i = 2:n
    c = find(~bitget(uint32(i-2), 1:B), 1);
    x = bitxor(x, V(:,c)');
    X(i,:) = double(bitxor(x, shift)) / 2^32;
end
end
