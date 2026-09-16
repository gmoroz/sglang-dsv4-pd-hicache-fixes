# SGLang DeepSeek-V4 PD HiCache Fixes

This public fork contains engine-only fixes for DeepSeek-V4 prefill/decode (PD)
deployments, based on the upstream `v0.5.16` tag. It deliberately excludes
deployment wrappers, container files, Helm charts, and environment-specific
configuration.

## What is fixed

- **Per-pool HiCache metrics.** The fork exports
  `sglang:hicache_host_pool_used_tokens{pool=...}` and
  `sglang:hicache_host_pool_total_tokens{pool=...}` for every host pool in a
  `HostPoolGroup`. The standard host metrics describe only the FULL pool and can
  hide a nearly-full SWA pool.
- **Metrics cannot terminate the scheduler.** A pool whose `available_size()`
  is intentionally unsupported (for example, a state sidecar without an
  allocator) is skipped. Other failures are logged and isolated from the
  scheduler loop instead of propagating into process termination.
- **Correct host eviction wiring.** C4-state and C4-indexer-state sidecars do
  not receive an eviction callback because they have no `alloc`/`free` pair.
  C4, C4-indexer, and C128 component pools evict through the FULL component so
  host space can be reserved for transfers.
- **Allocation diagnostics.** The diagnostic branch adds throttled
  `LOADBACK_GATE` and `LOADBACK_PATH` log points around host load-back decisions.
- **Coordinated FULL/SWA host eviction.** Per-component eviction could drop the
  SWA host copy of a radix node while keeping its FULL copy, leaving an entry
  that no longer produced a useful host hit. Eviction now walks the SWA host LRU,
  skips locked and stale nodes, and releases the paired FULL and SWA copies
  together. A separate `hicache_swa_ratio` sizes the SWA host pool independently
  of `hicache_ratio`, which still sizes the state pools.
- **Tool calls with a completed body but no outer marker.** The DeepSeek and
  Qwen3-Coder detectors accept a function body whose inner invoke marker is
  closed even when the outer marker is absent; a truncated body without a
  closing marker is still rejected. When a tool-call marker is present but the
  detector returns neither a call nor normal text, the parser keeps the original
  text instead of dropping the response silently.
- **Degenerate repetition guard.** A request whose output collapses into a
  repeating tail is finished deterministically: the scheduler checks the last
  320 tokens for a period of 4 to 64 tokens repeated five times, at most once
  per 32 accepted tokens. Covered by unit tests for the period bounds, the
  repeat threshold, and non-repeating output.

These changes address invisible per-pool saturation, failed host-space
reservation, scheduler crashes caused by unsupported state-sidecar metrics,
host copies that survive their pair, dropped tool calls, and requests that
never terminate on repeated output.

## Upstream context

The relevant upstream fixes are not included in this fork. Related discussion
is tracked in [issue #35129](https://github.com/sgl-project/sglang/issues/35129),
[issue #33713](https://github.com/sgl-project/sglang/issues/33713),
[PR #34046](https://github.com/sgl-project/sglang/pull/34046), and
[PR #31771](https://github.com/sgl-project/sglang/pull/31771).

## Branches

- `dsv4-hicache-fixes` contains the production fixes described above.
- `dsv4-hicache-diagnostics` contains the same fixes plus the load-back
  diagnostic log points.
