# Changelog

## 0.1.1 - 2026-05-06

- Added a live CrewAI tier-2 integration test that builds a real one-agent Crew
  and verifies end-to-end MNEMOS MCP tool dispatch during `crew.kickoff()`.

## 0.1.0 - 2026-05-04

- Added the initial CrewAI adapter for MNEMOS MCP tools.
- Added dynamic CrewAI `BaseTool` subclass generation from MCP JSON Schema inputs.
- Added offline unit tests for schema conversion, tool generation, and adapter wiring.
- Added a live integration test gated by MNEMOS environment variables.
