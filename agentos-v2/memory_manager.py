import json
import os
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage

MEMORY_FILE = Path(os.getcwd()) / "agentos_memory.json"
MAX_MEMORY_TOKENS = int(os.getenv("MEMORY_TOKEN_BUDGET", "512"))

COMPRESSION_PROMPT = """You are a memory compressor. Given the action taken and its result, \
update the memory tokens to preserve the most important state for future steps. \
Be extremely concise. Use structured notation. Preserve: file paths edited, \
test results, errors encountered, plan remaining, key decisions made. \
Output ONLY the updated memory block in JSON format.

CURRENT MEMORY:
{current_memory_json}

NEW ACTION:
Tool called: {tool_name}
Input: {tool_input_summary}
Result: {tool_result_summary}

OUTPUT FORMAT (JSON only):
{{
  "files_edited": ["path:lines_changed"],
  "tests": {{"passed": 0, "failed": 0, "failing_tests": []}},
  "plan_remaining": ["step1", "step2"],
  "key_decisions": ["decision1"],
  "errors_seen": ["error_type: brief_description"],
  "context_summary": "one sentence about overall task state"
}}"""

def load_memory() -> dict:
    """Load persistent memory state from disk, or return empty schema."""
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "files_edited": [],
        "tests": {"passed": 0, "failed": 0, "failing_tests": []},
        "plan_remaining": [],
        "key_decisions": [],
        "errors_seen": [],
        "context_summary": "No prior context."
    }

def save_memory(memory: dict) -> None:
    """Save persistent memory state to disk."""
    MEMORY_FILE.write_text(json.dumps(memory, indent=2), encoding="utf-8")

def memory_to_system_message(memory: dict) -> SystemMessage:
    """Convert memory dictionary to a SystemMessage prepended to every LLM call."""
    tests = memory.get('tests', {})
    content = f"""
[PERSISTENT MEMORY — {MAX_MEMORY_TOKENS} token budget]
Files edited this session: {', '.join(memory.get('files_edited', [])) or 'none'}
Tests: {tests.get('passed', 0)} passed / {tests.get('failed', 0)} failed
Failing: {', '.join(tests.get('failing_tests', [])) or 'none'}
Plan remaining: {' → '.join(memory.get('plan_remaining', [])) or 'task complete'}
Key decisions: {'; '.join(memory.get('key_decisions', [])) or 'none'}
Errors seen: {'; '.join(memory.get('errors_seen', [])) or 'none'}
State: {memory.get('context_summary', 'No context')}
[END MEMORY]
""".strip()
    return SystemMessage(content=content)

async def update_memory(
    current: dict,
    tool_name: str,
    tool_input: str,
    tool_result: str,
    compressor_llm
) -> dict:
    """
    Run the compression step after a tool call.
    Uses the compressor LLM to merge the new tool result into the existing memory dict.
    """
    prompt = COMPRESSION_PROMPT.format(
        current_memory_json=json.dumps(current, indent=2),
        tool_name=tool_name,
        tool_input_summary=str(tool_input)[:300],
        tool_result_summary=str(tool_result)[:500],
    )
    try:
        response = await compressor_llm.ainvoke([HumanMessage(content=prompt)])
        # Extract JSON from potential markdown blocks
        content = getattr(response, 'content', str(response)).strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
        
        updated = json.loads(content)
        save_memory(updated)
        return updated
    except Exception:
        # If compression fails (parsing error, API error), return the current memory safely
        return current
