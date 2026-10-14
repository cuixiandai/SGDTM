import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def index_reverse(index):
    B, N = index.shape
    ind = torch.arange(N, device=index.device).expand(B, N)
    index_r = torch.empty_like(index)
    index_r.scatter_(dim=1, index=index, src=ind)
    return index_r


def _sgn_gather(x, index):
    B, C, H, W = x.shape
    N = H * W
    assert index.shape[0] == B
    assert index.shape[-1] == N

    x_flat = x.reshape(B, C, N)
    index_exp = index.unsqueeze(1).expand(B, C, N)
    out = torch.gather(x_flat, dim=2, index=index_exp)
    return out.reshape(B, C, H, W)


def unfold(x, index):
    return _sgn_gather(x, index)


def fold(x, index_reverse):
    return _sgn_gather(x, index_reverse)


def _auto_num_tokens(dim, ratio=1.5, min_tokens=16, max_tokens=256):
    t = max(min_tokens, min(int(round(dim * ratio)), max_tokens))
    return 2 ** int(round(math.log2(t)))


class StructureRouter(nn.Module):
    def __init__(self, dim, num_tokens=None, auto_ratio=1.5):
        super().__init__()
        if num_tokens is None:
            num_tokens = _auto_num_tokens(dim, ratio=auto_ratio)
        self.num_tokens = num_tokens

        self.route = nn.Sequential(
            nn.Linear(dim, dim // 3),
            nn.GELU(),
            nn.Linear(dim // 3, num_tokens),
            nn.LogSoftmax(dim=-1),
        )
        self.prompt_gate = nn.Parameter(torch.zeros(num_tokens))

    def forward(self, x, tau=1.0, hard=True):
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.reshape(B, C, N).permute(0, 2, 1).contiguous()  # (B, N, C)

        logits = self.route(x_flat)                                # (B, N, T)
        logits = logits + torch.log(F.softplus(self.prompt_gate) + 1e-6)
        cls_policy = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        labels = torch.argmax(cls_policy.detach(), dim=-1)         # (B, N)
        return labels, cls_policy


def build_indices(labels):
    _, sort_index = torch.sort(labels, dim=-1, stable=False)
    sort_index_reverse = index_reverse(sort_index)
    return sort_index, sort_index_reverse


if __name__ == '__main__':
    torch.manual_seed(42)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    B, C, H, W = 2, 64, 13, 13
    x = torch.randn(B, C, H, W, device=device)

    router = StructureRouter(dim=C).to(device)
    labels, _ = router(x)
    sort_index, sort_index_reverse = build_indices(labels)

    x_unfolded = unfold(x, sort_index)
    x_folded = fold(x_unfolded, sort_index_reverse)

    print(f"device:          {device}")
    print(f"Input:            {tuple(x.shape)}")
    print(f"Unfold Input: {tuple(x_unfolded.shape)}")
    print(f"Fold output:   {tuple(x_folded.shape)}")
