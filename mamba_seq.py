"""
Stage-2 ablation: bidirectional selective state-space model (Mamba-style)
across slices.

A linear-time recurrent model whose state transition depends on the input
(the Mamba S6 selective scan), an alternative both to the BiGRU's gated
recurrence and to attention. Written in plain PyTorch (a chunked scan,
see chunked_selective_scan), so it needs no mamba_ssm/CUDA extension. Layout
matches bigru_lstm.py: 768 -> 256 projection, 2 bidirectional layers,
attention pooling and the shared regression head. There is no positional
encoding, because the scan is order-aware on its own.
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from stage2_common import AttentionPooling, RegressionHead, seed_from_env

STAGE2_SEEDABLE = True


class SelectiveSSM(nn.Module):
    """One direction of a Mamba block (in_proj, causal depthwise conv, selective scan, gate)."""
    def __init__(self, d_model, d_state=16, expand=2, d_conv=4, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner, self.d_state = d_inner, d_state
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # S4D-real initialisation of A and the standard Mamba dt bias init
        A = torch.arange(1, d_state + 1, dtype=torch.float).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus

    def forward(self, x):
        # x: [B, L, d_model]
        L = x.size(1)
        u, z = self.in_proj(x).chunk(2, dim=-1)                       # [B, L, d_inner] each
        u = F.silu(self.conv1d(u.transpose(1, 2))[..., :L]).transpose(1, 2)

        dt, B, C = self.x_proj(u).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                             # [B, L, d_inner]
        A = -torch.exp(self.A_log)                                    # [d_inner, d_state]

        log_dA = dt.unsqueeze(-1) * A                                 # [B, L, d_inner, d_state], <= 0
        dBu = dt.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)     # [B, L, d_inner, d_state]
        y = _scan_fn(log_dA.device)(log_dA, dBu, C) + u * self.D       # [B, L, d_inner]
        return self.out_proj(y * F.silu(z))


def chunked_selective_scan(log_dA, dBu, C, chunk=8, min_log_decay=-5.0):
    """Computes y_t = C_t . h_t with h_t = exp(log_dA_t) * h_{t-1} + dBu_t (h_0 = 0).

    Within a chunk, with S_t = sum_{j<=t} log_dA_j (<= 0), the recurrence has
    the closed form h_t = exp(S_t) * cumsum_k(exp(-S_k) * dBu_k), so a chunk
    costs a few elementwise ops and one cumsum instead of a [chunk x chunk]
    decay matrix. Only the state carried between chunks is looped
    (L/chunk iterations).

    Numerics: exp(-S) is bounded by exp(-min_log_decay * chunk) = e^40 for the
    defaults, because each per-step log-decay is floored at min_log_decay. The
    bound must stay well below fp32 overflow after multiplying by the largest
    |dBu| (trained models reach ~1e8, so keep chunk * |min_log_decay| <= ~40;
    chunk 16 with e^80 overflowed to NaN on real features). The floor is part
    of the model definition: it only affects steps that already forget more
    than 99.3% of the state, and on trained weights it moved predictions by
    ~0.002 years on average versus the unfloored sequential recurrence.
    log_dA, dBu: [B, L, D, N]; C: [B, L, N]. Returns [B, L, D].
    """
    Bsz, L, D, N = dBu.shape
    log_dA = log_dA.clamp(min=min_log_decay)
    pad = (-L) % chunk
    if pad:  # zero padding at the end cannot affect earlier outputs
        log_dA = F.pad(log_dA, (0, 0, 0, 0, 0, pad))
        dBu = F.pad(dBu, (0, 0, 0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, pad))
    nc = (L + pad) // chunk
    a = log_dA.view(Bsz, nc, chunk, D, N)
    b = dBu.view(Bsz, nc, chunk, D, N)
    c = C.view(Bsz, nc, chunk, N)

    S = a.cumsum(dim=2)                                               # [B, nc, Q, D, N], <= 0
    expS = torch.exp(S)
    h_intra = expS * torch.cumsum(torch.exp(-S) * b, dim=2)          # chunk-local states (zero initial state)

    # carry the state across chunk boundaries
    chunk_decay = expS[:, :, -1]                                      # [B, nc, D, N]
    h = dBu.new_zeros(Bsz, D, N)
    carried = []
    for ci in range(nc):
        carried.append(h)
        h = chunk_decay[:, ci] * h + h_intra[:, ci, -1]
    h_all = h_intra + expS * torch.stack(carried, dim=1).unsqueeze(2)

    y = torch.einsum('bctdn,bctn->bctd', h_all, c).reshape(Bsz, nc * chunk, D)
    return y[:, :L]


_compiled_scan = None


def _scan_fn(device):
    """chunked_selective_scan, compiled with torch.compile on CUDA (~2.5x faster
    end to end at batch size 1). Set MAMBA_COMPILE=0 to force eager mode; any
    compile failure falls back to eager."""
    global _compiled_scan
    if device.type != 'cuda' or os.environ.get('MAMBA_COMPILE', '1') == '0':
        return chunked_selective_scan
    if _compiled_scan is None:
        try:
            _compiled_scan = torch.compile(chunked_selective_scan)
        except Exception as e:  # pragma: no cover
            print(f"[mamba_seq] torch.compile unavailable ({e}); using eager scan")
            _compiled_scan = chunked_selective_scan
    return _compiled_scan


class BiMambaLayer(nn.Module):
    """Pre-norm residual layer that runs one SSM forward and one backward over the slices."""
    def __init__(self, d_model, dropout, **ssm_kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = SelectiveSSM(d_model, **ssm_kwargs)
        self.bwd = SelectiveSSM(d_model, **ssm_kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        out = self.fwd(h) + self.bwd(h.flip(1)).flip(1)
        return x + self.dropout(out)


class AgePredictionCNN(nn.Module):
    def __init__(self, input_shape, hidden_dim=256, num_layers=2, d_state=16, dropout=0.3):
        seed_from_env()
        super().__init__()
        print("AgePredictionCNN (bidirectional Mamba SSM) Shape:", input_shape)

        self.num_slices = input_shape[1]
        self.embedding_dim = input_shape[2]

        self.input_projection = nn.Linear(self.embedding_dim, hidden_dim)
        self.layers = nn.ModuleList([
            BiMambaLayer(hidden_dim, dropout=0.1, d_state=d_state) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention_pool = AttentionPooling(hidden_dim)
        self.head = RegressionHead(hidden_dim, dropout=dropout)

    def _encode(self, x):
        x = self.input_projection(x.squeeze(1))  # [B, S, hidden]
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def forward(self, x, sex):
        x, _ = self.attention_pool(self._encode(x))
        return self.head(x, sex)

    def get_attention_weights(self, x, sex):
        """Slice-pooling weights for visualisation, [batch, num_slices]."""
        _, weights = self.attention_pool(self._encode(x))
        return weights.detach()
