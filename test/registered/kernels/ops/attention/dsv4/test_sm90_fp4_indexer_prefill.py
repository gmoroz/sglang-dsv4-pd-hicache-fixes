"""Fused prefill indexer scores: values and top-k selection vs the torch scoring,
the length mask, and the torch prefill loop with the kernel switched on and off."""

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.sm90_fp4_indexer_prefill import (
    fused_index_scores,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

HEADS = 32
HEAD_DIM = 128
TOPK = 512
MIN_OVERLAP = 0.99
MIN_EXACT = 0.99
MAX_REL_ERR = 1e-2


def _torch_scores(q, k, weights):
    """`DeepseekV41Indexer.scores`: bf16 up to the head reduction."""
    s = torch.einsum("bhd,nd->bhn", q, k)
    return (s.relu() * weights.unsqueeze(-1)).sum(dim=1).float()


def _inputs(rows, lc, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(rows, HEADS, HEAD_DIM, generator=g, device="cuda").bfloat16()
    k = torch.randn(lc, HEAD_DIM, generator=g, device="cuda").bfloat16()
    weights = torch.randn(rows, HEADS, generator=g, device="cuda").bfloat16()
    # Prefill tail of a prompt: row i sees lc - rows + 1 + i positions.
    lens = torch.arange(lc - rows + 1, lc + 1, device="cuda").clamp_min(1)
    return q, k, weights, lens


def _mask(scores, lens):
    j = torch.arange(scores.shape[1], device=scores.device)
    return scores.masked_fill(j[None, :] >= lens[:, None], -torch.inf)


def _topk_overlap(a, b, k):
    ia, ib = a.topk(k, dim=-1).indices, b.topk(k, dim=-1).indices
    hits = (ia.unsqueeze(-1) == ib.unsqueeze(-2)).any(-1).sum(-1)
    return (hits.float() / k).mean().item()


@pytest.mark.parametrize("rows,lc", [(1, 700), (77, 1000), (300, 4099), (513, 20011)])
def test_fused_index_scores_random_inputs_match_torch_scoring(rows, lc):
    q, k, weights, lens = _inputs(rows, lc, seed=rows)
    out = fused_index_scores(q, k, weights, lens)
    ref = _mask(_torch_scores(q, k, weights), lens)

    assert out.dtype == torch.float32 and out.shape == (rows, lc)
    assert torch.equal(out.isneginf(), ref.isneginf())
    visible = ref.isfinite()
    # Same bf16 rounding points as torch: values differ only by head summation
    # order, i.e. at most one bf16 step on a few entries.
    exact = (out == ref)[visible].float().mean().item()
    assert exact >= MIN_EXACT, f"bit-equal fraction {exact}"
    rel_err = (out - ref)[visible].abs().max() / ref[visible].abs().max()
    assert rel_err.item() <= MAX_REL_ERR
    overlap = _topk_overlap(out, ref, min(TOPK, int(lens.min())))
    assert overlap >= MIN_OVERLAP, f"top-{TOPK} overlap {overlap}"


class _Indexer:
    """Precomputed queries and head weights behind the indexer interface."""

    def __init__(self, q, weights):
        self.q, self.w = q, weights
        self.index_topk = TOPK
        self.is_candidate_source = False
        self.uses_candidates = False

    def queries(self, q_lora, freqs):
        return self.q[q_lora]

    def head_weights(self, x):
        return self.w[x]

    def scores(self, q, k, weights):
        return _torch_scores(q, k, weights)


def _run_torch_prefill(fused, q, k, weights, ratio, seq_len, oracle=False):
    from sglang.srt.layers.attention.deepseek_v4_backend import (
        DeepseekV4AttnBackend,
        _low_ratio_sparse_buffers,
    )

    rows = q.shape[0]
    pos = torch.arange(seq_len - rows, seq_len, device="cuda")
    lens = ((pos + 1) // ratio).clamp_min(1).to(torch.int32)
    _, page_indices, raw_indices = _low_ratio_sparse_buffers(
        lens, TOPK, is_prefill=True
    )
    backend = object.__new__(DeepseekV4AttnBackend)
    backend.forward_metadata = SimpleNamespace(
        core_metadata=SimpleNamespace(
            sparse_page_indices=lambda r: page_indices,
            sparse_raw_indices=lambda r: raw_indices,
        )
    )
    # Identity page table: compressed slot of position j is j.
    backend.req_to_token = torch.arange(seq_len, device="cuda")[None].int()
    backend.token_to_kv_pool = SimpleNamespace(
        get_low_ratio_index_k_dequant=lambda layer_id, slots: k[slots]
    )
    backend.candidate_masks = None
    layer = SimpleNamespace(
        layer_id=0,
        compress_ratio=ratio,
        indexer=_Indexer(q, weights),
        freqs_cis=torch.zeros(seq_len, device="cuda"),
    )
    tok = torch.arange(rows, device="cuda")
    req = torch.zeros(rows, dtype=torch.int64, device="cuda")
    with (
        envs.SGLANG_OPT_USE_FUSED_DSV41_INDEXER_SCORES.override(fused),
        envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.override(oracle),
    ):
        backend._low_ratio_index_topk_torch(layer, tok, tok, req, pos)
    return page_indices[:rows], raw_indices[:rows], (pos + 1) // ratio


@pytest.mark.parametrize("ratio", [1, 2])
def test_torch_prefill_fused_scores_select_like_torch_scoring(ratio):
    seq_len, rows = 3001, 600
    q, k, weights, _ = _inputs(rows, seq_len // ratio, seed=ratio)
    fused_pages, fused_raw, lens = _run_torch_prefill(
        True, q, k, weights, ratio, seq_len
    )
    torch_pages, torch_raw, _ = _run_torch_prefill(False, q, k, weights, ratio, seq_len)

    n_valid = lens.clamp_max(TOPK)
    j = torch.arange(fused_raw.shape[1], device="cuda")
    selected = j[None, :] < n_valid[:, None]
    assert torch.equal(fused_raw >= 0, selected)
    assert bool((fused_raw < lens[:, None]).all())
    assert torch.equal(fused_pages, fused_raw)
    hits = (
        (fused_raw.unsqueeze(-1) == torch_raw.unsqueeze(-2)).any(-1) & selected
    ).sum(-1)
    overlap = (hits / n_valid).mean().item()
    assert overlap >= MIN_OVERLAP, f"{ratio=}: overlap {overlap}"


def test_torch_prefill_oracle_flag_skips_fused_kernel(monkeypatch):
    """The torch prefill indexer flag is the oracle for the dense fp4 kernel: it
    must score with torch even when the fused kernel is enabled."""
    from sglang.srt.layers.attention import deepseek_v4_backend

    def _fail(*args, **kwargs):
        raise AssertionError("fused kernel ran under the torch oracle flag")

    monkeypatch.setattr(deepseek_v4_backend, "fused_index_scores", _fail)
    seq_len, rows = 1001, 64
    q, k, weights, _ = _inputs(rows, seq_len, seed=3)
    _, oracle_raw, _ = _run_torch_prefill(True, q, k, weights, 1, seq_len, oracle=True)
    assert bool((oracle_raw[:, :TOPK] >= 0).all())
