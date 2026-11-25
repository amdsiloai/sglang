"""ROCm coverage for the QSA packed-varlen decode fallback, the FlyDSL sparse
prefill and indexer scoring.

On HIP there is neither flash_attn (FA2) nor the FA4 CuTe kernels, so
``_resolve_flash_attn_varlen_func`` resolves to aiter's FA2-compatible varlen
kernel. With SGLANG_ROCM_QSA_FLYDSL_PREFILL on gfx950, QSA extend attention runs
on aiter's FlyDSL sparse prefill (ROCm/aiter#7556). Tests with fake modules are
device-independent; the others run the real aiter kernels against a torch
reference and only execute on ROCm. The indexer tests check the grouped verify
scoring on gfx950, the rows per request the indexer passes it for verify and
draft-extend graphs, and the draft-extend graph layout the grouping relies on.
"""

import sys
from itertools import accumulate
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa_backend_module
from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.runner_utils.prefill_graph import prefill_graph_scope
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.mem_cache_utils import routed

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=60, stage="base-a-jit-kernel-unit", runner_config="amd")


def _fake_aiter(varlen_func):
    module = ModuleType("aiter")
    module.flash_attn_varlen_func = varlen_func
    return module


@pytest.mark.parametrize("returns_tuple", [False, True], ids=["tensor", "tuple"])
def test_qsa_hip_resolves_aiter_varlen_kernel(monkeypatch, returns_tuple):
    resolver = qsa_backend_module._resolve_flash_attn_varlen_func
    resolver.cache_clear()
    sentinel = object()
    calls = []

    def aiter_varlen(*args, **kwargs):
        calls.append(kwargs)
        return (sentinel, None) if returns_tuple else sentinel

    monkeypatch.setattr("sglang.srt.utils.is_sm121", lambda: False)
    monkeypatch.setattr("sglang.srt.utils.is_hip", lambda: True)
    monkeypatch.setitem(sys.modules, "aiter", _fake_aiter(aiter_varlen))

    try:
        func = resolver()
        assert func(q=1, causal=True) is sentinel
        assert calls == [{"q": 1, "causal": True}]
    finally:
        resolver.cache_clear()


def test_qsa_hip_without_aiter_falls_through_to_flash_attn(monkeypatch):
    resolver = qsa_backend_module._resolve_flash_attn_varlen_func
    resolver.cache_clear()
    fa2_func = object()
    flash_attn = ModuleType("flash_attn")
    flash_attn.flash_attn_varlen_func = fa2_func

    monkeypatch.setattr("sglang.srt.utils.is_sm121", lambda: False)
    monkeypatch.setattr("sglang.srt.utils.is_hip", lambda: True)
    monkeypatch.setitem(sys.modules, "aiter", None)  # makes `import aiter` fail
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)

    try:
        assert resolver() is fa2_func
    finally:
        resolver.cache_clear()


def test_qsa_hip_aiter_varlen_matches_torch_reference():
    if not (is_hip() and torch.cuda.is_available()):
        pytest.skip("ROCm-only kernel")
    pytest.importorskip("aiter")

    resolver = qsa_backend_module._resolve_flash_attn_varlen_func
    resolver.cache_clear()
    torch.manual_seed(2028)
    device = torch.device("cuda")
    # Qwen3.8-Flash-Next full-attention shape: 24 query heads, 2 KV heads, d=256.
    batch, topk = 3, 2048
    num_q_heads, num_kv_heads, head_dim = 24, 2, 256
    valid_counts = torch.tensor([5, topk, 700], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu_seqlens_k[1:] = torch.cumsum(valid_counts, 0)
    cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=device)
    q = torch.randn(batch, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    packed_k = torch.randn(
        int(valid_counts.sum()),
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    packed_v = torch.randn_like(packed_k)
    scale = head_dim**-0.5

    try:
        # Mirrors the decode call in QwenSparseAttnBackend._forward_paged_attention.
        out = resolver()(
            q=q,
            k=packed_k,
            v=packed_v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=1,
            max_seqlen_k=topk,
            softmax_scale=scale,
            causal=True,
        )
    finally:
        resolver.cache_clear()

    assert out.shape == (batch, num_q_heads, head_dim)
    repeats = num_q_heads // num_kv_heads
    for row in range(batch):
        start, end = int(cu_seqlens_k[row]), int(cu_seqlens_k[row + 1])
        keys = packed_k[start:end].float().repeat_interleave(repeats, dim=1)
        values = packed_v[start:end].float().repeat_interleave(repeats, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys) * scale
        expected = torch.einsum("hk,khd->hd", scores.softmax(dim=-1), values)
        torch.testing.assert_close(out[row].float(), expected, atol=2e-2, rtol=2e-2)


def test_qsa_flydsl_prefill_needs_the_aiter_api(monkeypatch):
    # An aiter.ops.qsa from before ROCm/aiter#7556 has the op but not its caller API.
    old_qsa = ModuleType("aiter.ops.qsa")
    old_qsa.qsa_sparse_prefill = object()
    monkeypatch.setitem(sys.modules, "aiter", ModuleType("aiter"))
    monkeypatch.setitem(sys.modules, "aiter.ops", ModuleType("aiter.ops"))
    monkeypatch.setitem(sys.modules, "aiter.ops.qsa", old_qsa)
    monkeypatch.setattr(qsa_backend_module, "is_gfx95_supported", lambda: True)
    monkeypatch.setenv("SGLANG_ROCM_QSA_FLYDSL_PREFILL", "1")
    resolver = qsa_backend_module._resolve_flydsl_sparse_prefill
    resolver.cache_clear()
    try:
        assert resolver() is None
    finally:
        resolver.cache_clear()


@pytest.mark.parametrize(
    "query_lens, prefix_lens",
    [((1536,), (0,)), ((1536, 700), (3000, 0))],
    ids=["no-prefix", "prefix"],
)
def test_qsa_extend_flydsl_sparse_prefill_matches_reference(
    monkeypatch, query_lens, prefix_lens
):
    if not (is_hip() and torch.cuda.is_available() and is_gfx95_supported()):
        pytest.skip("gfx950-only kernel")
    qsa = pytest.importorskip("aiter.ops.qsa")
    if not hasattr(qsa, "qsa_sparse_prefill_problem"):
        pytest.skip("aiter without ROCm/aiter#7556")

    torch.manual_seed(7556)
    device = torch.device("cuda")
    # Qwen3.8-Flash-Next: 24 query heads, 2 KV heads, d=256, 512 blocks of 4 tokens.
    heads, kv_heads, dim, ratio, token_topk = 24, 2, 256, 4, 2048
    block_topk = token_topk // ratio
    scale = dim**-0.5
    lengths = [n + p for n, p in zip(query_lens, prefix_lens)]
    rows = sum(query_lens)
    # Each request's K/V at random slots of a BF16 pool (slot 0 unused).
    slots = torch.randperm(sum(lengths), device=device, dtype=torch.int32) + 1
    req_to_token = torch.zeros(
        len(lengths) + 1, max(lengths), dtype=torch.int32, device=device
    )
    for r, (start, n) in enumerate(zip(accumulate(lengths, initial=0), lengths)):
        req_to_token[r + 1, :n] = slots[start : start + n]
    pool_k = torch.randn(
        sum(lengths) + 1, kv_heads, dim, device=device, dtype=torch.bfloat16
    )
    pool_v = torch.randn_like(pool_k)
    requests = torch.repeat_interleave(
        torch.arange(len(lengths), device=device),
        torch.tensor(query_lens, device=device),
    )
    positions = torch.cat(
        [p + torch.arange(n, device=device) for n, p in zip(query_lens, prefix_lens)]
    )
    new_slots = req_to_token[requests + 1, positions].long()
    q = torch.randn(rows, heads, dim, device=device, dtype=torch.bfloat16)

    # Random top-512 blocks among each row's complete ones, expanded by SGLang.
    visible_blocks = ((positions + 1) // ratio)[:, None]
    columns = torch.arange(max(lengths) // ratio, device=device)
    scores = torch.rand(rows, columns.numel(), device=device)
    scores = scores.masked_fill(columns >= visible_blocks, -1.0)
    take = min(block_topk, columns.numel())
    blocks = torch.full((rows, block_topk), -1, dtype=torch.int32, device=device)
    blocks[:, :take] = scores.topk(take, dim=1).indices.int()
    blocks[torch.arange(block_topk, device=device) >= visible_blocks] = -1
    indices = expand_qsa_block_indices(
        blocks,
        positions,
        torch.tensor(lengths, device=device)[requests],
        ratio,
        token_topk,
    )

    class Pool:
        def get_key_buffer(self, layer_id):
            return pool_k

        def get_value_buffer(self, layer_id):
            return pool_v

    backend = qsa_backend_module.QwenSparseAttnBackend()
    backend.token_to_kv_pool = routed(Pool())
    backend.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
    layer = SimpleNamespace(
        tp_q_head_num=heads, head_dim=dim, layer_id=0, scaling=scale
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        attn_cp_metadata=None,
        extend_seq_lens_cpu=list(query_lens),
        seq_lens_cpu=lengths,
        extend_seq_lens=torch.tensor(query_lens, dtype=torch.int32, device=device),
        req_pool_indices=torch.arange(1, len(lengths) + 1, device=device),
    )
    returned = []
    op = qsa.qsa_sparse_prefill
    monkeypatch.setattr(
        qsa,
        "qsa_sparse_prefill",
        lambda *a, **kw: returned.append(op(*a, **kw)) or returned[-1],
    )

    def extend(flag):
        monkeypatch.setenv("SGLANG_ROCM_QSA_FLYDSL_PREFILL", flag)
        qsa_backend_module._resolve_flydsl_sparse_prefill.cache_clear()
        try:
            out = backend.forward_extend(
                q.reshape(rows, -1),
                pool_k[new_slots],
                pool_v[new_slots],
                layer,
                batch,
                save_kv_cache=False,
                topk_indices=indices,
            )
        finally:
            qsa_backend_module._resolve_flydsl_sparse_prefill.cache_clear()
        return out.view(rows, heads, dim).float()

    flydsl = extend("1")
    assert len(returned) == 1 and torch.equal(flydsl, returned[0].float())
    triton = extend("0")
    assert len(returned) == 1
    # Prefill graph runs and calls the op refuses keep Triton.
    with prefill_graph_scope(full_graph=False):
        extend("1")
    assert len(returned) == 1
    monkeypatch.setattr(qsa, "qsa_sparse_prefill_problem", lambda *a, **kw: "no")
    extend("1")
    assert len(returned) == 1

    # float32 attention of sampled rows over the tokens their indices select.
    sampled = torch.linspace(0, rows - 1, 64, device=device).long().unique()
    expected = []
    for row in sampled.tolist():
        selected = indices[row][indices[row] >= 0].long()
        kv_slots = req_to_token[requests[row] + 1, selected].long()
        k, v = pool_k[kv_slots].float(), pool_v[kv_slots].float()
        q_row = q[row].float().view(kv_heads, heads // kv_heads, dim)
        p = torch.softmax(torch.einsum("hgd,nhd->hgn", q_row, k) * scale, dim=-1)
        expected.append(torch.einsum("hgn,nhd->hgd", p, v).reshape(heads, dim))
    expected = torch.stack(expected)
    # MI355X: at most 0.0038 from the reference (Triton: 0.0044), 0.0156 from Triton.
    assert (flydsl[sampled] - expected).abs().max() <= 1e-2
    assert (flydsl - triton).abs().max() <= 5e-2


@pytest.mark.parametrize("requests", [1, 7, 16, 32])
@pytest.mark.parametrize("min_scored_len", [0, 512])
@pytest.mark.parametrize("padded_heads", [8, 16])
def test_qsa_gfx950_grouped_verify_scoring_is_bitwise(
    requests, min_scored_len, padded_heads
):
    """The indexer scores the four verify rows of a request from one key load on
    gfx950; every written logit must equal the per-row kernel's bit for bit."""
    from sglang.srt.layers.attention.qsa.mqa import (
        HAS_TILELANG,
        tilelang_qsa_mqa_decode,
    )
    from sglang.srt.utils import is_gfx95_supported

    if not (HAS_TILELANG and is_gfx95_supported()):
        pytest.skip("gfx950 TileLang-only path")
    torch.manual_seed(requests)
    device = torch.device("cuda")
    rows, max_pages, page_size = 4 * requests, 256, 16
    max_model_len = max_pages * page_size
    # Verify rows of a request share its pages and end one token apart.
    lens = torch.randint(1, max_model_len - 3, (requests,)).repeat_interleave(4)
    lens = (lens + torch.arange(4).repeat(requests)).to(device, torch.int32)
    page_table = torch.randint(0, 1024, (requests, max_pages), dtype=torch.int32)
    page_table = page_table.repeat_interleave(4, dim=0).to(device)
    cache = torch.randn(1024, page_size, 1, 128, device=device).to(torch.bfloat16)
    # The indexer projects 4 heads, zero-padded to the kernel's head alignment.
    q = torch.zeros(rows, padded_heads, 128, device=device, dtype=torch.bfloat16)
    q[:, :4] = torch.randn(rows, 4, 128, device=device).to(torch.bfloat16)

    args = (q, cache, page_table, lens, max_model_len)
    ref = tilelang_qsa_mqa_decode(*args, min_scored_len=min_scored_len)
    out = tilelang_qsa_mqa_decode(
        *args, min_scored_len=min_scored_len, rows_per_request=4, num_query_heads=4
    )
    for row in range(rows):
        length = int(lens[row])
        if length <= min_scored_len:
            continue
        assert torch.equal(
            out[row, :length].view(torch.int32), ref[row, :length].view(torch.int32)
        )


class _ForwardMode:
    def __init__(self, mode):
        self.mode = mode

    def is_decode(self):
        return self.mode == "decode"

    def is_target_verify(self):
        return self.mode == "verify"

    def is_draft_extend_v2(self):
        return self.mode == "draft_extend"


@pytest.mark.parametrize(
    "mode, graph, expected",
    [
        ("draft_extend", True, 4),
        ("draft_extend", False, 1),
        ("verify", True, 4),
        ("verify", False, 1),
        ("decode", True, 1),
    ],
)
def test_qsa_decode_rows_per_request(mode, graph, expected):
    """Verify and draft-extend graphs pass their per-request row count to the
    decode scorer; eager batches and plain decode score row by row."""
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

    rows, seen = 8, []

    def select_decode_tokens(*args, rows_per_request, **kwargs):
        seen.append(rows_per_request)

    indexer = SimpleNamespace(
        layer_id=0,
        index_n_heads=4,
        project_qk=lambda *args, **kwargs: (torch.zeros(rows, 8, 128), None, True),
        update_key_state_and_compress=lambda *args, **kwargs: None,
        select_decode_tokens=select_decode_tokens,
    )
    metadata = SimpleNamespace(
        is_cuda_graph=graph,
        defer_block_expansion=False,
        token_to_kv_pool=None,
        decode_logical_positions=torch.arange(rows),
        pending_ring_slots=torch.zeros(rows, dtype=torch.int64),
        get_token_to_batch_idx=lambda: torch.arange(rows) // 4,
        get_decode_mqa_inputs=lambda layer_id: (None, None, None, 1),
        get_seqlens_int32=lambda: torch.ones(rows, dtype=torch.int32),
    )
    # Each mode's spec info carries only its own field: the draft-extend window
    # width (and logprob rows), or verify's draft_token_num. Decode carries both.
    spec_info = {
        "draft_extend": SimpleNamespace(
            num_tokens_per_req=4, num_tokens_for_logprob_per_req=1
        ),
        "verify": SimpleNamespace(draft_token_num=4),
        "decode": SimpleNamespace(num_tokens_per_req=4, draft_token_num=4),
    }[mode]
    batch = SimpleNamespace(forward_mode=_ForwardMode(mode), spec_info=spec_info)
    QSAIndexer._forward_impl(
        indexer,
        torch.zeros(rows, 16),
        torch.zeros(rows, dtype=torch.int64),
        batch,
        metadata,
    )
    assert seen == [expected]


def test_qsa_draft_extend_graph_layout_groups_rows_by_request():
    """The draft-extend graph layout the grouped decode scorer reads: each request's
    window rows back to back on its request slot, the last row the longest. Request
    0 is shorter than the window; the last two are the graph runners' padding
    requests (slot 0, length 1)."""
    from sglang.srt.layers.attention.qsa.graph_metadata import (
        _qsa_graph_layout_kernel,
    )

    if not torch.cuda.is_available():
        pytest.skip("Triton kernel")
    torch.manual_seed(0)
    device = torch.device("cuda")
    width, bs, padding = 4, 12, 2
    rows = bs * width
    # Post-extend lengths and request slots.
    seq_lens = torch.randint(width, 70000, (bs,), device=device)
    req_pool = torch.randperm(bs, device=device) + 1
    seq_lens[0] = 2
    seq_lens[-padding:] = 1
    req_pool[-padding:] = 0
    extend_lens = torch.full((bs,), width, dtype=torch.int32, device=device)
    row_seq_lens, row_prefix_lens, row_req_pool = (
        torch.full((rows,), -1, dtype=torch.int32, device=device) for _ in range(3)
    )
    _qsa_graph_layout_kernel[(bs + 1,)](
        seq_lens,
        req_pool,
        extend_lens,
        row_seq_lens,
        row_prefix_lens,
        row_req_pool,
        bs,
        rows,
        0,
        num_warps=1,
    )
    groups = row_seq_lens.view(bs, width)
    assert torch.equal(
        row_req_pool.view(bs, width), req_pool[:, None].expand(bs, width).int()
    )
    assert bool((groups.diff(dim=1) >= 0).all())
    assert torch.equal(groups[:, -1], seq_lens.int())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
