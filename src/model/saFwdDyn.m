function ddq = saFwdDyn(p, q, dq, tau)
%SAFWDDYN 解析モデルの前進動力学: M(q)·ddq = τ − g(q) − C(q,dq)dq − (Bv·dq + Fc·tanh(dq/ε))
%   ddq = SAFWDDYN(p, q, dq, tau)   p: c8_params。q, dq, tau は [nAx x 1]（nAx ≤ 6）。
%   可動でない軸（nAx+1..6）は 0 rad で固定。c8RNEA は armature を含み、摩擦は含まない。
n = numel(q);  z6 = zeros(6,1);
qq = [q(:); zeros(6-n,1)];  dd = [dq(:); zeros(6-n,1)];
g = c8RNEA(qq, z6, z6, p);
M = zeros(6);
for j = 1:6, e = z6;  e(j) = 1;  M(:,j) = c8RNEA(qq, z6, e, p) - g; end
b = c8RNEA(qq, dd, z6, p) - g;                       % コリオリ・遠心
fr = p.frictionBv(:).*dd + p.frictionFc(:).*tanh(dd./p.frictionEps(:));
ddq = M(1:n,1:n) \ (tau(:) - g(1:n) - b(1:n) - fr(1:n));
end
