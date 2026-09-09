# Changelog

## Unreleased

### Security

- Added a read-only-by-default CrewAI tool policy backed by an explicit allowlist
  copied from the canonical MNEMOS server contract. Unknown tool names now fail
  closed instead of bypassing a mutating-verb denylist. Callers can still opt in
  with `allowed_tools` or restore unrestricted exposure with `readonly=False`.

### Fixed

- Classified the server contract's `log_memory` and `checkout_memory` tools as
  read-only and included them in the default allowlist.

## 0.1.1 - 2026-05-06

- Added a live CrewAI tier-2 integration test that builds a real one-agent Crew
  and verifies end-to-end MNEMOS MCP tool dispatch during `crew.kickoff()`.

## 0.1.0 - 2026-05-04

- Added the initial CrewAI adapter for MNEMOS MCP tools.
- Added dynamic CrewAI `BaseTool` subclass generation from MCP JSON Schema inputs.
- Added offline unit tests for schema conversion, tool generation, and adapter wiring.
- Added a live integration test gated by MNEMOS environment variables.
