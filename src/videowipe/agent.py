"""Optional local coding-agent integration for intent selection."""
from __future__ import annotations

from dataclasses import dataclass
import errno
import json
import shlex
import shutil
import subprocess
from typing import Iterable

from videowipe.detect import CleanCandidate


_AGENT_COMMANDS = {
    "codex": ["codex", "exec", "-"],
    "claude": ["claude", "-p"],
    "gemini": ["gemini", "-p"],
}


@dataclass(frozen=True)
class AgentSelectionResult:
    """Selection plus safe diagnostic codes, never subprocess text or prompts."""

    status: str
    selected_ids: tuple[str, ...] = ()
    reason: str | None = None

    def diagnostic(self) -> dict:
        return {
            "status": self.status,
            "selection_source": "agent" if self.status == "success" else "local_rules",
            "reason": self.reason,
        }


def select_with_agent(
    agent: str,
    candidates: Iterable[CleanCandidate],
    intent: str,
    timeout: int = 30,
) -> list[str] | None:
    """Compatibility API: return IDs (including ``[]``), or None on failure."""
    result = select_with_agent_result(agent, candidates, intent, timeout)
    return list(result.selected_ids) if result.status == "success" else None


def select_with_agent_result(
    agent: str,
    candidates: Iterable[CleanCandidate],
    intent: str,
    timeout: float = 30,
) -> AgentSelectionResult:
    """Ask a local agent and distinguish failures without exposing its output."""
    candidate_list = list(candidates)
    try:
        command = _resolve_agent_command(agent)
    except ValueError:
        return AgentSelectionResult("command_unavailable", reason="invalid_command_syntax")
    if command is None:
        return AgentSelectionResult("command_unavailable", reason="command_not_found")

    prompt = _build_prompt(candidate_list, intent)
    try:
        proc = subprocess.run(
            command,
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return AgentSelectionResult("timeout", reason="deadline_exceeded")
    except OSError as exc:
        if exc.errno in {errno.ENOENT, errno.EACCES, errno.ENOEXEC}:
            return AgentSelectionResult("command_unavailable", reason="command_not_executable")
        return AgentSelectionResult("execution_failed", reason="process_start_failed")
    if proc.returncode != 0:
        return AgentSelectionResult("execution_failed", reason="nonzero_exit")

    try:
        output = proc.stdout.decode("utf-8") if isinstance(proc.stdout, bytes) else proc.stdout
    except UnicodeError:
        return AgentSelectionResult("invalid_output", reason="invalid_encoding")
    selected = _parse_agent_ids(output, {candidate.id for candidate in candidate_list})
    if selected is None:
        return AgentSelectionResult("invalid_output", reason="invalid_selection")
    return AgentSelectionResult("success", tuple(selected))


def _resolve_agent_command(agent: str) -> list[str] | None:
    key = agent.strip()
    if not key:
        return None
    command = _AGENT_COMMANDS.get(key.lower())
    if command is None:
        command = shlex.split(key)
    if not command or shutil.which(command[0]) is None:
        return None
    return command


def _build_prompt(candidates: list[CleanCandidate], intent: str) -> str:
    payload = {
        "intent": intent,
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
    return (
        "You are selecting video cleanup targets. "
        "Return only compact JSON in this exact shape: "
        "{\"remove\":[\"candidate-id\"]}. "
        "Do not explain. Candidates and user intent:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def _parse_agent_ids(output: str, valid_ids: set[str]) -> list[str] | None:
    text = output.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Retain compatibility with fenced JSON and CLI explanatory wrappers.
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            return None
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    remove = data.get("remove")
    if not isinstance(remove, list):
        return None
    selected: list[str] = []
    for item in remove:
        if not isinstance(item, str) or item not in valid_ids:
            return None
        if item not in selected:
            selected.append(item)
    return selected
