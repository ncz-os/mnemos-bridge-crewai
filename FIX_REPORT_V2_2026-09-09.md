# MNEMOS CrewAI Bridge B1/M1 Fix Report V2

Date: 2026-09-09

Repository: `/tmp/mnemos-audit-clones/mnemos-bridge-crewai`

Branch: `fix/audit-findings-v2`

Base: `master` at `cc6968f` (`fix(packaging): repair the build-system table on the hatchling projects`)

Remote action: none; no fetch, clone, or push was performed

## Outcome

Finding B1 and Finding M1 are fixed.

The default CrewAI tool-exposure policy is now a true, fail-closed allowlist.
Only names explicitly present in `READ_ONLY_TOOL_NAMES` are exposed when
`readonly=True`. An unrecognized server-advertised name is withheld regardless
of whether its words resemble a known mutating verb.

`log_memory` and `checkout_memory` are now explicitly classified as read-only,
matching the MNEMOS server contract.

## Server-contract evidence

The source of truth inspected for the policy was:

- `/Users/jasonperlow/Projects/mnemos/mnemos/mcp/tools/__init__.py`
- local repository commit: `5c756437a195ffe2e14934d49222b905e5d34d4e`
- file SHA-256: `828fed9e5dbf51dd995f3658d0e08beb992eca423f31940de7d85081ac003027`

The same file hash was present in two other current local MNEMOS checkouts.
The registry contains 25 tool names and identifies these eight as writes:

- `branch_memory`
- `bulk_create_memories`
- `create_memory`
- `delete_memory`
- `delete_triple`
- `kg_create_triple`
- `update_memory`
- `update_triple`

The bridge allowlist is the exact 17-name set difference between the registry
and `_WRITE_TOOLS`:

- `checkout_memory`
- `diff_memory_commits`
- `get_memory`
- `get_stats`
- `graeae_consult`
- `graeae_get_consultation`
- `kg_search`
- `kg_timeline`
- `kronos_anomalies`
- `kronos_forecast`
- `list_deletions`
- `list_memories`
- `log_memory`
- `pantheon_list_models`
- `pantheon_route_explain`
- `recommend_model`
- `search_memories`

An AST-based comparison reported:

```text
server_tools 25
server_write_tools 8
expected_read_only 17
bridge_allowlist 17
missing_from_bridge []
extra_in_bridge []
exact_match True
```

## Finding B1 remediation

Added `src/mnemos_bridge_crewai/tool_policy.py` with an immutable explicit
`READ_ONLY_TOOL_NAMES` set copied from the real server contract.

The default behavior is now:

1. If `allowed_tools` is supplied, expose only those exact names. This is an
   explicit opt-in and may include writes.
2. Otherwise, if `readonly=True`, expose only names in
   `READ_ONLY_TOOL_NAMES`.
3. Apply `denied_tools` last as a subtractive policy.
4. If `readonly=False`, expose all advertised tools except names explicitly
   denied.

Unknown names therefore fail closed by default. The regression fixture makes
the fake server advertise the invented `grant_memory_access` tool and verifies
that `crewai_tools()` does not return it. A second invented name,
`rotate_api_key`, is covered by the policy classification test.

## Finding M1 remediation

`log_memory` and `checkout_memory` are members of the explicit read-only
allowlist. Tests now assert both:

- `is_read_only_tool_name(name)` is true; and
- `is_destructive_tool_name(name)` is false.

The adapter-level default-policy test also advertises both tools and verifies
that both are returned while write and invented names are withheld.

## Adapter behavior

`MnemosCrewAIAdapter.connect()` and `crewai_tools()` now accept `readonly`,
`allowed_tools`, and `denied_tools` policy arguments. Per-call arguments can
override the adapter defaults.

The adapter retains all server-advertised schemas in `tool_schemas` for
inspection, but only builds CrewAI tool instances for permitted schemas. Tool
classes and instances are cached by name, the cache is invalidated when the
dispatcher changes, duplicate advertised names are reduced to the first schema
with a warning, and the default read-only call warns once with the withheld
names.

The current `master` implementation of schema conversion, synchronous
dispatch, no-retry mutation handling, and structured MCP error propagation was
preserved. The older `origin/fix/audit-findings` tool-factory implementation was
not transplanted over newer `master` code.

## Files changed

- `src/mnemos_bridge_crewai/tool_policy.py`
- `src/mnemos_bridge_crewai/adapter.py`
- `src/mnemos_bridge_crewai/__init__.py`
- `tests/test_adapter_offline.py`
- `README.md`
- `CHANGELOG.md`
- `FIX_REPORT_V2_2026-09-09.md`

## Verification

### Focused tests

Command:

```text
/private/tmp/mnemos-probe.KSNyXD/.venv/bin/pytest tests/test_adapter_offline.py -q
```

Result:

```text
22 passed in 0.09s
```

### Full repository test suite

Command:

```text
/private/tmp/mnemos-probe.KSNyXD/.venv/bin/pytest -q -ra
```

Result:

```text
34 passed, 2 skipped in 0.13s
```

The two skips were the live integration tests:

- `tests/integration/test_crewai_crew.py`: `MNEMOS_TEST_BASE` was not set.
- `tests/integration/test_crewai_crew_kickoff.py`: the live MNEMOS endpoint,
  token, and LLM-key requirements were not satisfied.

No test failed.

### Formatting and lint

Ruff 0.16.6 ran with `check --fix` and `format` on every changed Python file.
The resulting changed-file check contains three findings, all present in the
`master` versions of the same code:

- `PYI034` on `MnemosCrewAIAdapter.__aenter__`.
- `PYI034` on `FakeMcpClient.__aenter__`.
- `SIM117` on the pre-existing nested context managers in the timeout test.

There are zero new Ruff findings. Import-order findings present on `master`
were fixed by the formatter. `git diff --check` passes.

### Review gate

Zoder was attempted first, as required. Its authoring invocation could not start
because the local zeroclaw daemon socket did not appear. Its review invocation
then reached the selected reviewer but the provider request failed; zero of one
reviewers completed. That incomplete result was not treated as an approval.

The fallback secondary Codex adversarial review completed read-only with:

```text
APPROVE
No blocking findings.
```

The reviewer independently confirmed the exact contract match, fail-closed
handling of unknown names, correct M1 classifications, policy precedence,
warning behavior, instance reuse, dispatcher-sensitive cache invalidation, and
regression coverage. It made no file, Git, or remote changes.

## Environment note

`uv sync --extra dev` could not download `pytest-asyncio` because this sandbox
could not resolve PyPI. It created only an ignored local `.venv`. The real
repository suite was therefore run with an existing local Python 3.13 test
environment containing pytest 9.1.1, pytest-asyncio 1.4.0, Pydantic 2.13.5, and
Ruff 0.16.6. The live integration requirements remained unavailable and were
reported as skips above.

## Commit and push status

All scoped changes, including this report, are intended for one local
conventional commit authored as `Jason Perlow <jperlow@gmail.com>`. The exact
commit hash is reported alongside this file after the commit is created because
a commit cannot accurately contain its own hash.

No push was performed. The user retains responsibility for pushing the branch.
