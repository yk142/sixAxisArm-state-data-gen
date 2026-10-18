function p = saApplyPayload(p, payload)
%SAAPPLYPAYLOAD 手先リンク（arm6 = p.links(7)）に点質量のペイロードを合成する（issue #61）。
%   p = SAAPPLYPAYLOAD(p, payload)   payload.mass [kg]、payload.offset [3 x 1]（arm6 座標系での位置 [m]。既定 [0 0 0.05]）
%   質量・重心・重心まわりの慣性（リンク座標系）を平行軸定理で合成する。mass = 0 または空なら何もしない。
%   親資産の c8_params / writeC8URDF / c8RNEA は変更せず、パラメータ構造体 p だけを書き換える。
if isempty(payload) || ~isfield(payload,'mass') || payload.mass == 0, return; end
off = [0; 0; 0.05];  if isfield(payload,'offset') && ~isempty(payload.offset), off = payload.offset(:); end
L = p.links(7);  c0 = L.com(:);  m0 = L.mass;  m1 = payload.mass;
m = m0 + m1;  c = (m0*c0 + m1*off)/m;
I = L.inertiaC;                                    % c0 まわり（リンク座標系）
for pair = {{m0, c0}, {m1, off}}                  % 平行軸定理で合成重心 c まわりへ。ペイロードは点質量（自身の慣性なし）
    mi = pair{1}{1};  d = pair{1}{2} - c;
    I = I + mi*((d'*d)*eye(3) - d*d');
end
p.links(7).mass = m;  p.links(7).com = c(:)';  p.links(7).inertiaC = I;
end
