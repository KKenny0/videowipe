"""Optional local coding-agent integration for intent selection."""
from __future__ import annotations

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


def select_with_agent(
    agent: str,
    candidates: Iterable[CleanCandidate],
    intent: str,
    timeout: int = 30,
) -> list[str] | None:
    """Ask a local agent CLI which candidate ids should be removed.

    Returns ``None`` when the agent is unavailable or returns invalid output.
    """
    candidate_list = list(candidates)
    command = _resolve_agent_command(agent)
    if command is None:
        return None

    prompt = _build_prompt(candidate_list, intent)
    try:
        proc = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None

    return _parse_agent_ids(proc.stdout, {candidate.id for candidate in candidate_list})


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
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
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
