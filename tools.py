import wikipedia
from typing import Optional
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from database import AsyncSessionLocal, Task

repl = PythonREPL()


@tool
def python_execution_tool(code: str) -> str:
    """Codex Engine: Executes Python code for math and logic validation."""
    try:
        return repl.run(code)
    except Exception as e:
        return f"Execution Error: {repr(e)}"


async def _async_db_add(description: str, due_date: Optional[str]):
    async with AsyncSessionLocal() as db:
        try:
            new_task = Task(description=description, due_date=due_date)
            db.add(new_task)
            await db.commit()
            return f"DB Agent: Successfully retained context for '{description}'."
        except Exception as e:
            await db.rollback()
            return f"DB Error: {str(e)}"


@tool
async def add_database_task(description: str, due_date: Optional[str] = None) -> str:
    """DB Agent: Stores information for long-term project memory."""
    return await _async_db_add(description, due_date)


@tool
def research_tool(query: str) -> str:
    """Research Agent: Fetches documentation and technical summaries."""
    try:
        return wikipedia.summary(query, sentences=4)
    except Exception:
        return f"Research failed for '{query}'."

