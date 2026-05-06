from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import os
import signal
from typing import Any, Iterator

import pytest

from mnemos_bridge_crewai import MnemosCrewAIAdapter


REQUIRED_ENV_VARS = ("MNEMOS_TEST_BASE", "MNEMOS_MCP_TOKEN")
LLM_KEY_ENV_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
RESULT_TERMS = ("memory", "infrastructure", "mnemos")


pytestmark = pytest.mark.skipif(
    not all(os.environ.get(name) for name in REQUIRED_ENV_VARS)
    or not any(os.environ.get(name) for name in LLM_KEY_ENV_VARS),
    reason=(
        "requires MNEMOS_TEST_BASE, MNEMOS_MCP_TOKEN, and either "
        "OPENAI_API_KEY or ANTHROPIC_API_KEY"
    ),
)


def test_one_agent_crewai_crew_kickoff_dispatches_live_mnemos_tool() -> None:
    with _deadline(seconds=30):
        crewai = pytest.importorskip("crewai")
        _run_live_crew_kickoff(crewai)


def _run_live_crew_kickoff(crewai: Any) -> None:
    adapter = asyncio.run(
        MnemosCrewAIAdapter.connect(
            os.environ["MNEMOS_TEST_BASE"],
            os.environ["MNEMOS_MCP_TOKEN"],
            timeout=8,
        )
    )

    recorder = RecordingDispatcher(adapter._dispatcher)
    adapter._dispatcher = recorder

    try:
        tools = asyncio.run(adapter.crewai_tools())
        assert tools, "Live MNEMOS endpoint did not expose any MCP tools."

        tool_names = [str(tool.name) for tool in tools]
        agent = crewai.Agent(
            role="MNEMOS researcher",
            goal="search MNEMOS for relevant context about a query",
            backstory=(
                "You retrieve context from MNEMOS using the MCP tools available "
                "to you before producing a concise answer."
            ),
            tools=tools,
            llm=_build_llm(crewai),
            cache=False,
            verbose=False,
            max_iter=3,
            max_execution_time=24,
            allow_delegation=False,
        )
        task = crewai.Task(
            description=(
                "Use an MNEMOS MCP tool before answering. Search MNEMOS for "
                "memories about infrastructure, then summarize the findings in "
                "2 sentences. Include the words MNEMOS and infrastructure in "
                "the final answer. Do not write, update, or delete memories. "
                f"Available MNEMOS MCP tools include: {', '.join(tool_names)}."
            ),
            expected_output=(
                "A 2 sentence summary of relevant MNEMOS infrastructure memories."
            ),
            agent=agent,
        )
        crew = crewai.Crew(
            agents=[agent],
            tasks=[task],
            cache=False,
            memory=False,
            verbose=False,
        )

        result = crew.kickoff()
        result_text = _result_to_text(result)
        evidence = _collect_tool_use_evidence(
            crew=crew,
            agent=agent,
            task=task,
            tools=tools,
            result_text=result_text,
            recorder=recorder,
        )

        assert evidence, (
            "Expected CrewAI to expose or emit evidence of at least one tool "
            f"invocation. Result was: {result_text[:500]!r}"
        )
        assert recorder.calls, (
            "Expected the CrewAI agent to dispatch at least one live MNEMOS MCP "
            f"tool call. CrewAI evidence: {evidence!r}. "
            f"Result was: {result_text[:500]!r}"
        )
        assert result_text.strip(), "Crew kickoff returned an empty result."
        assert any(term in result_text.lower() for term in RESULT_TERMS), (
            "Crew result did not mention memory, infrastructure, or MNEMOS. "
            f"Result was: {result_text[:500]!r}"
        )
    finally:
        asyncio.run(adapter.aclose())


def _build_llm(crewai: Any) -> Any:
    if os.environ.get("OPENAI_API_KEY"):
        return crewai.LLM(
            model="gpt-4o-mini",
            provider="openai",
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0,
            timeout=15,
            max_tokens=300,
        )

    return crewai.LLM(
        model="claude-haiku-4-5",
        provider="anthropic",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        temperature=0,
        timeout=15,
        max_tokens=300,
    )


def _result_to_text(result: Any) -> str:
    raw = getattr(result, "raw", None)
    if raw:
        return str(raw)

    if hasattr(result, "model_dump"):
        dumped = result.model_dump()
        if isinstance(dumped, dict) and dumped.get("raw"):
            return str(dumped["raw"])

    return str(result)


def _collect_tool_use_evidence(
    *,
    crew: Any,
    agent: Any,
    task: Any,
    tools: list[Any],
    result_text: str,
    recorder: RecordingDispatcher,
) -> list[str]:
    evidence: list[str] = []

    if recorder.calls:
        evidence.append(
            "dispatcher.calls="
            + ",".join(call.name for call in recorder.calls)
        )

    used_tools = getattr(task, "used_tools", 0)
    if used_tools:
        evidence.append(f"task.used_tools={used_tools}")

    tools_results = getattr(agent, "tools_results", None)
    if tools_results:
        used_names = [
            str(item.get("tool_name"))
            for item in tools_results
            if isinstance(item, dict) and item.get("tool_name")
        ]
        evidence.append(f"agent.tools_results={used_names or len(tools_results)}")

    last_run = getattr(agent, "last_run", None)
    if last_run:
        evidence.append("agent.last_run=true")

    tasks_output = getattr(crew, "tasks_output", None)
    if tasks_output:
        evidence.append(f"crew.tasks_output={len(tasks_output)}")

    usage_metrics = getattr(crew, "usage_metrics", None)
    if usage_metrics is not None:
        evidence.append("crew.usage_metrics=true")

    result_lower = result_text.lower()
    mentioned_tool_names = [
        str(tool.name)
        for tool in tools
        if str(tool.name).lower() in result_lower
    ]
    if mentioned_tool_names:
        evidence.append(f"result.tool_mentions={mentioned_tool_names}")

    return evidence


@dataclass
class RecordedCall:
    name: str
    args: dict[str, Any]


class RecordingDispatcher:
    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        self.calls: list[RecordedCall] = []

    async def call(self, name: str, args: dict[str, Any]) -> str:
        self.calls.append(RecordedCall(name=name, args=dict(args)))
        return await self._wrapped.call(name, args)


@contextmanager
def _deadline(*, seconds: int) -> Iterator[None]:
    if not hasattr(signal, "setitimer"):
        yield
        return

    def raise_timeout(signum: int, frame: Any) -> None:
        raise TimeoutError(f"live CrewAI integration test exceeded {seconds} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])
