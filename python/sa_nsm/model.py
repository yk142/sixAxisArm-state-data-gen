"""Neural State Model（1 ステップ遷移）。A: 純粋 MLP、B: 解析モデル＋NN の残差、C: ゲート付きの 2 項残差（issue #35）。

入力・出力はすべて HDF5 の spec 単位:
  X_spec = [sin q, cos q, dq/DQ_MAX, tau/TAU_MAX]（3 軸で 12 列）、Y_spec = [Δq/D_Q_MAX, Δdq/D_DQ_MAX]（6 列）
解析モデルの予測 ana（spec 単位、6 列）とリミット距離特徴 lim（6 列）は、データ側で事前計算して渡す。
"""
import math

import torch
import torch.nn as nn

LIM_UNIT = math.radians(1.0)         # リミット距離特徴の単位（1°）


def limit_features(q, q_min, q_max):
    """リミットまでの距離（1° 単位）。[(qMax − q), (q − qMin)] の 2n 列。範囲外は負、遠い側は 10 で打ち切る。"""
    return torch.clamp(torch.cat([(q_max - q), (q - q_min)], -1) / LIM_UNIT, -3.0, 10.0)


def mlp(d_in, d_out, width=256, depth=3, zero_last=False):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    last = nn.Linear(d, d_out)
    if zero_last:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class Base(nn.Module):
    kind = "base"

    def __init__(self, n_ax, stats, width=256, depth=3):
        super().__init__()
        self.n_ax, self.width, self.depth = n_ax, width, depth
        for k, v in stats.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))

    def config(self):
        return {"kind": self.kind, "n_ax": self.n_ax, "width": self.width, "depth": self.depth}


class PureNSM(Base):
    """A: 入力 X_spec（標準化）→ Y_spec（標準化）。stats: x_mean, x_std, y_mean, y_std。"""
    kind = "A"

    def __init__(self, n_ax, stats, width=256, depth=3):
        super().__init__(n_ax, stats, width, depth)
        self.net = mlp(4 * n_ax, 2 * n_ax, width, depth)

    def forward(self, xs, ana, lim):
        return self.net((xs - self.x_mean) / self.x_std) * self.y_std + self.y_mean

    def loss(self, pred, y, at_limit, lim=None):
        return (((pred - y) / self.y_std) ** 2).mean()


class ResidualNSM(Base):
    """B: 次状態 = 解析モデル + NN の残差。入力は [X_spec, ana, lim]（標準化）、出力は残差 / σ_all。最終層はゼロ初期化。
    stats: x_mean, x_std（4n+2n+2n 列）, sig_all（2n）。"""
    kind = "B"

    def __init__(self, n_ax, stats, width=256, depth=3):
        super().__init__(n_ax, stats, width, depth)
        self.net = mlp(8 * n_ax, 2 * n_ax, width, depth, zero_last=True)

    def features(self, xs, ana, lim):
        return (torch.cat([xs, ana, lim], -1) - self.x_mean) / self.x_std

    def forward(self, xs, ana, lim):
        return ana + self.net(self.features(xs, ana, lim)) * self.sig_all

    def loss(self, pred, y, at_limit, lim=None):
        return (((pred - y) / self.sig_all) ** 2).mean()


class GatedResidualNSM(ResidualNSM):
    """C: 次状態 = 解析モデル + σ_free·h_free(z) + g(q)·σ_contact·h_contact(z)。
    g = max_a sigmoid((0.5 − d_a)/0.2)（d_a は軸 a のリミットまでの距離 [°]）。接触は他軸にも波及するので、ゲートは全出力に共通。
    損失は、遷移の区分（at_limit）ごとの σ で標準化する。stats: x_mean, x_std, sig_free, sig_contact。"""
    kind = "C"

    def __init__(self, n_ax, stats, width=256, depth=3, width_c=None):
        Base.__init__(self, n_ax, stats, width, depth)
        self.width_c = width_c or max(64, width // 2)
        self.net = mlp(8 * n_ax, 2 * n_ax, width, depth, zero_last=True)                   # 非接触の項
        self.net_c = mlp(8 * n_ax, 2 * n_ax, self.width_c, max(2, depth - 1), zero_last=True)   # 接触の項

    def config(self):
        return {**super().config(), "width_c": self.width_c}

    def gate(self, lim):
        n = self.n_ax
        d = torch.minimum(lim[:, :n], lim[:, n:])
        return torch.sigmoid((0.5 - d) / 0.2).amax(-1, keepdim=True)

    def forward(self, xs, ana, lim):
        z = self.features(xs, ana, lim)
        return ana + self.net(z) * self.sig_free + self.gate(lim) * self.net_c(z) * self.sig_contact

    def zone(self, lim, at_limit):
        """接触ゾーン = at_limit またはゲートが開いている（> 0.01）遷移。
        at_limit は σ_contact、ゾーン内の非接触は σ_free と σ_contact の幾何平均、ゾーン外は σ_free で標準化する（リミット近傍で接触に入るかは入力だけでは決まらず、
        σ_free で測るとゾーン内の接触項のわずかな出力に 10⁷ 倍の罰則が掛かるため）。"""
        return at_limit | (self.gate(lim).squeeze(-1) > 0.01)

    def loss(self, pred, y, at_limit, lim=None):
        if lim is None:
            sig = torch.where(at_limit.unsqueeze(-1), self.sig_contact, self.sig_free)
        else:       # at_limit は σ_contact、ゾーン内の非接触は幾何平均、ゾーン外は σ_free
            mid = torch.sqrt(self.sig_free * self.sig_contact)
            sig = torch.where(at_limit.unsqueeze(-1), self.sig_contact, torch.where(self.zone(lim, at_limit).unsqueeze(-1), mid, self.sig_free))
        return (((pred - y) / sig) ** 2).mean()

    def train_loss(self, xs, ana, lim, y, at_limit):
        return self.loss(self(xs, ana, lim), y, at_limit, lim)


MODELS = {"A": PureNSM, "B": ResidualNSM, "C": GatedResidualNSM}
