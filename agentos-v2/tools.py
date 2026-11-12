"""
tools.py — AgentOS tool definitions.

CodexEngine   → execute_code, analyze_code, read_file, list_directory,
                write_file, patch_file, search_codebase,
                run_tests, run_shell_command, index_repo
ResearchAgent → web_search, scrape_page, search_algorithms
DBAgent       → save_code_snippet, get_snippets, log_improvement
"""

import ast
import asyncio
import difflib
import json
import logging
import os
import pathlib
import re
import subprocess
from typing import Optional

import requests

log = logging.getLogger("agentos.tools")
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_experimental.utilities import PythonREPL
from sqlalchemy import select

from database import AsyncSessionLocal, CodeSnippet, Improvement

repl = PythonREPL()

# ── Path safety ─────────────────────────────────────────────────────────────────
# Resolve paths relative to CWD; block traversal outside the project root.
_PROJECT_ROOT = pathlib.Path(os.getcwd()).resolve()

def _safe_resolve(path: str) -> pathlib.Path:
    """Resolve `path` relative to the project root, rejecting escape attempts."""
    resolved = (_PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(_PROJECT_ROOT)
    except ValueError:
        raise PermissionError(f"Path '{path}' escapes the project root — refused.")
    return resolved


# ══════════════════════════════════════════════════════════════════════════════
# CODEX ENGINE TOOLS
# ══════════════════════════════════════════════════════════════════════════════


@tool
async def execute_code(code: str) -> str:
    """Execute Python code in a sandbox and return its stdout / result.
    Use to validate solutions, test logic, or run examples BEFORE recommending
    them. Prefer running code over guessing the output.
    Input: raw Python source code (not a shell command)."""
    try:
        result = await asyncio.to_thread(repl.run, code)
        return result.strip() if result and result.strip() else "(executed — no output)"
    except Exception as e:
        return f"Execution Error: {repr(e)}"


@tool
async def analyze_code(code: str) -> str:
    """Static-analyse Python code using the AST.
    Detects: syntax errors, bare excepts, mutable default args, eval/exec,
    missing docstrings. Reports line/function/class counts.
    Use before recommending refactors or when the user shares code for review."""
    issues: list[str] = []
    suggestions: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax Error (line {e.lineno}): {e.msg}\n  -> {e.text}"

    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            issues.append(
                f"Line {node.lineno}: Bare `except:` — specify the exception type."
            )
        if isinstance(node, ast.FunctionDef):
            for default in node.args.defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    issues.append(
                        f"Line {node.lineno}: Mutable default arg in `{node.name}()` — "
                        "use None and initialise inside the body."
                    )
            has_doc = (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
            )
            if not has_doc:
                suggestions.append(f"Line {node.lineno}: `{node.name}()` has no docstring.")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec"):
                issues.append(f"Line {node.lineno}: `{node.func.id}()` is a security risk.")

    lc = len(code.splitlines())
    fc = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    cc = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    parts = [f"Stats: {lc} lines | {fc} functions | {cc} classes"]
    parts.append(
        "Issues:\n" + "\n".join(f"  - {i}" for i in issues)
        if issues else "No critical issues detected."
    )
    if suggestions:
        parts.append("Suggestions:\n" + "\n".join(f"  - {s}" for s in suggestions))
    return "\n\n".join(parts)


@tool
async def read_file(path: str) -> str:
    """Read a file and return its content with line numbers.
    Use to understand existing code before modifying it.
    Always read a file before editing it."""
    try:
        resolved = _safe_resolve(path)
        if not resolved.is_file():
            return f"Error: '{path}' is not a file or does not exist."
        content = await asyncio.to_thread(resolved.read_text, encoding="utf-8")
        lines = content.splitlines()
        numbered = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
        return f"File: {resolved}  ({len(lines)} lines)\n{'─'*60}\n{numbered}"
    except Exception as e:
        return f"Error reading file: {repr(e)}"


@tool
async def list_directory(path: str = ".") -> str:
    """List files and subdirectories at the given path (tree style with sizes).
    Use to explore project structure before reading or modifying files."""
    try:
        resolved = _safe_resolve(path)
        if not resolved.is_dir():
            return f"Error: '{path}' is not a directory."
        entries = []
        for item in sorted(resolved.iterdir()):
            if item.name.startswith(".") or item.name in (
                "__pycache__", "node_modules", ".venv", ".git", ".mypy_cache"
            ):
                continue
            if item.is_dir():
                n = sum(1 for _ in item.rglob("*") if _.is_file())
                entries.append(f"  📁 {item.name}/  ({n} files)")
            else:
                sz = item.stat().st_size
                sz_str = (f"{sz} B" if sz < 1024
                          else f"{sz/1024:.1f} KB" if sz < 1024**2
                          else f"{sz/1024**2:.1f} MB")
                entries.append(f"  📄 {item.name}  ({sz_str})")
        return (
            f"Directory: {resolved}\n{'─'*60}\n" + "\n".join(entries)
            if entries else f"'{resolved}' is empty."
        )
    except Exception as e:
        return f"Error listing directory: {repr(e)}"


@tool
async def write_file(path: str, content: str) -> str:
    """Create or fully overwrite a file with the given content.
    USE ONLY IN AUTONOMOUS MODE — when the developer explicitly says
    'create this file', 'implement this', 'write this to X'.
    To surgically modify an existing file use patch_file instead."""
    try:
        resolved = _safe_resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(resolved.write_text, content, encoding="utf-8")
        return f"Wrote {len(content.splitlines())} lines to {resolved}"
    except Exception as e:
        return f"Error writing file: {repr(e)}"


@tool
async def patch_file(path: str, old_lines: str, new_lines: str) -> str:
    """Surgically replace a block of lines in an existing file using a fuzzy
    unified-diff patch. Safer than raw string replace because it:
      • Strips trailing whitespace before matching (catches invisible space diffs)
      • Verifies the block appears EXACTLY ONCE (refuses ambiguous edits)
      • Shows a before/after diff in the return value for confirmation

    USE ONLY IN AUTONOMOUS MODE.

    Args:
        path:      File to edit (relative to project root)
        old_lines: The exact lines to replace (copy from read_file output,
                   omit the leading '   N | ' line-number prefix)
        new_lines: The replacement lines
    """
    try:
        resolved = _safe_resolve(path)
        if not resolved.is_file():
            return f"Error: '{path}' does not exist. Use write_file to create it."

        original = await asyncio.to_thread(resolved.read_text, encoding="utf-8")

        # Normalise: strip trailing whitespace per line for matching
        def _norm(text: str) -> str:
            return "\n".join(line.rstrip() for line in text.splitlines())

        norm_original = _norm(original)
        norm_old      = _norm(old_lines)

        occurrences = norm_original.count(norm_old)
        if occurrences == 0:
            # Build a helpful near-miss hint
            old_first = old_lines.splitlines()[0].strip() if old_lines.strip() else ""
            hint = ""
            if old_first:
                for i, line in enumerate(original.splitlines(), 1):
                    if old_first[:20].lower() in line.lower():
                        hint = f"\n  Nearest match found at line {i}: {line[:80]}"
                        break
            return (
                f"patch_file ERROR: old_lines not found in {resolved.name}.{hint}\n"
                f"Tip: call read_file('{path}') first and copy the exact lines."
            )
        if occurrences > 1:
            return (
                f"patch_file ERROR: old_lines appears {occurrences} times in "
                f"{resolved.name} — make old_lines longer/more specific to be unambiguous."
            )

        # Apply the patch on the normalised content, then reattach original line endings
        updated_norm = norm_original.replace(norm_old, _norm(new_lines), 1)

        # Restore the file's original line ending style
        eol = "\r\n" if "\r\n" in original else "\n"
        updated = updated_norm.replace("\n", eol)

        await asyncio.to_thread(resolved.write_text, updated, encoding="utf-8")

        # Show a compact diff for confirmation
        diff = list(difflib.unified_diff(
            original.splitlines(), updated.splitlines(),
            fromfile=f"a/{resolved.name}", tofile=f"b/{resolved.name}", lineterm=""
        ))
        diff_str = "\n".join(diff[:60]) + ("\n... (diff truncated)" if len(diff) > 60 else "")
        removed = len(old_lines.splitlines())
        added   = len(new_lines.splitlines())
        return (
            f"Patched {resolved.name}: -{removed} lines / +{added} lines\n"
            f"{'─'*60}\n{diff_str}"
        )
    except PermissionError as e:
        return f"Security Error: {e}"
    except Exception as e:
        return f"Error patching file: {repr(e)}"


@tool
async def search_codebase(pattern: str, path: str = ".", file_glob: str = "*.py") -> str:
    """Search the codebase for a regex pattern across all matching files.
    Returns each match with file path, line number, and surrounding context.
    Use to find: function definitions, usages of a variable, import statements,
    TODO comments, error handling patterns, or any symbol across the project.

    Args:
        pattern:   Python regex (e.g. 'def train', 'import torch', 'TODO')
        path:      Directory to search (default: project root)
        file_glob: File pattern to match (default: '*.py', use '*.js' for JS etc.)
    """
    try:
        resolved = _safe_resolve(path)
        regex    = re.compile(pattern, re.IGNORECASE)
        matches  = []

        for file in sorted(resolved.rglob(file_glob)):
            # Skip noise
            if any(p in file.parts for p in (".git", "__pycache__", ".venv", "node_modules")):
                continue
            try:
                lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue
            for i, line in enumerate(lines, 1):
                if regex.search(line):
                    context_start = max(0, i - 3)
                    context_end   = min(len(lines), i + 2)
                    ctx = "\n".join(
                        f"  {'>>>' if j + 1 == i else '   '} {j+1:4d} | {lines[j]}"
                        for j in range(context_start, context_end)
                    )
                    rel = file.relative_to(_PROJECT_ROOT)
                    matches.append(f"📄 {rel}:{i}\n{ctx}")

        if not matches:
            return f"No matches for `{pattern}` in {file_glob} files under '{path}'."
        header = f"Found {len(matches)} match(es) for `{pattern}`:\n{'═'*60}"
        return header + "\n\n" + "\n\n".join(matches[:30]) + (
            f"\n\n... {len(matches)-30} more matches (narrow your pattern)" if len(matches) > 30 else ""
        )
    except PermissionError as e:
        return f"Security Error: {e}"
    except re.error as e:
        return f"Invalid regex pattern: {e}"
    except Exception as e:
        return f"Error searching codebase: {repr(e)}"


@tool
async def run_tests(
    test_path: str = ".",
    pytest_args: str = "-v --tb=short",
) -> str:
    """Run the test suite with pytest and return structured results.
    Parses test output into: PASSED / FAILED / ERROR counts, individual
    failure details with tracebacks, and a NEXT ACTION recommendation.
    Use in a fix loop: run_tests → read failures → patch_file → run_tests again.

    Args:
        test_path:   File or directory to test (e.g. 'tests/', 'tests/test_api.py')
        pytest_args: Extra pytest flags (default: '-v --tb=short')
    """
    try:
        cmd = f"python -m pytest {test_path} {pytest_args} --no-header -q"

        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(_PROJECT_ROOT),
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            proc.kill()
            return "run_tests ERROR: pytest timed out after 120s — check for infinite loops or hung fixtures."

        out = stdout.decode(errors="replace")
        err = stderr.decode(errors="replace")
        full = (out + "\n" + err).strip()

        # Parse summary line: "5 passed, 2 failed, 1 error in 3.14s"
        n_passed = int(re.search(r"(\d+) passed", full).group(1)) if re.search(r"(\d+) passed", full) else 0
        failed_m = re.findall(r"(\d+) failed", full)
        error_m  = re.findall(r"(\d+) error", full)
        n_failed = int(failed_m[0]) if failed_m else 0
        n_error  = int(error_m[0])  if error_m  else 0

        all_green = proc.returncode == 0

        # Extract individual FAILED blocks
        failed_blocks = re.findall(r"(FAILED .+?)(?=\nFAILED |\nPASSED |\n=|\Z)", full, re.DOTALL)
        short_errors  = re.findall(r"(_{5,}.+?)(?=_{5,}|\Z)", full, re.DOTALL)

        status_icon = "✅" if all_green else "❌"
        header = (
            f"{status_icon} pytest results — "
            f"{n_passed} passed | {n_failed} failed | {n_error} errors\n"
            f"{'═'*60}"
        )

        sections = [header]

        if not all_green:
            if short_errors:
                sections.append("FAILURE DETAILS:\n" + "\n".join(short_errors[:5]))
            elif failed_blocks:
                sections.append("FAILED TESTS:\n" + "\n".join(failed_blocks[:5]))
            sections.append(
                "NEXT ACTION: Call patch_file to fix the first failing test, "
                "then call run_tests again to verify."
            )
        else:
            sections.append("All tests passing — no action required.")

        # Always include raw tail for context
        tail = "\n".join(full.splitlines()[-30:])
        sections.append(f"RAW OUTPUT (last 30 lines):\n{tail}")

        return "\n\n".join(sections)

    except FileNotFoundError:
        return "run_tests ERROR: pytest not found. Run: pip install pytest"
    except Exception as e:
        return f"run_tests ERROR: {repr(e)}"


@tool
async def index_repo(path: str = ".") -> str:
    """Build a structural index of the entire codebase.
    Returns: file tree, all Python class/function definitions with line numbers,
    import graph, and TODO/FIXME comments.
    Use at the START of any autonomous task to understand the project before touching files.
    Results are concise enough to fit in context; re-run if files have changed."""
    try:
        resolved = _safe_resolve(path)
        output   = []

        # ── File tree ────────────────────────────────────────────────────────
        tree_lines = [f"PROJECT ROOT: {resolved}"]
        py_files   = []
        for file in sorted(resolved.rglob("*")):
            if any(p in file.parts for p in (".git", "__pycache__", ".venv", "node_modules")):
                continue
            rel = file.relative_to(resolved)
            depth = len(rel.parts) - 1
            indent = "  " * depth
            if file.is_dir():
                tree_lines.append(f"{indent}📁 {file.name}/")
            else:
                sz = file.stat().st_size
                sz_str = f"{sz/1024:.1f}KB" if sz >= 1024 else f"{sz}B"
                tree_lines.append(f"{indent}📄 {file.name} ({sz_str})")
                if file.suffix == ".py":
                    py_files.append(file)
        output.append("FILE TREE\n" + "─"*40 + "\n" + "\n".join(tree_lines))

        # ── Python symbol index ──────────────────────────────────────────────
        symbol_lines = []
        todos        = []
        imports_map  = {}

        for pyfile in py_files:
            rel = pyfile.relative_to(resolved)
            try:
                src   = pyfile.read_text(encoding="utf-8", errors="replace")
                lines = src.splitlines()
                tree  = ast.parse(src)
            except Exception:
                continue

            file_symbols = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = [a.arg for a in node.args.args]
                    file_symbols.append(
                        f"  {'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}"
                        f"def {node.name}({', '.join(args)})  [line {node.lineno}]"
                    )
                elif isinstance(node, ast.ClassDef):
                    file_symbols.append(f"  class {node.name}  [line {node.lineno}]")

            # Imports
            file_imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    file_imports += [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    file_imports.append(node.module)
            if file_imports:
                imports_map[str(rel)] = list(dict.fromkeys(file_imports))

            # TODOs
            for i, line in enumerate(lines, 1):
                if re.search(r"TODO|FIXME|HACK|XXX|BUG", line, re.IGNORECASE):
                    todos.append(f"  {rel}:{i}  {line.strip()}")

            if file_symbols:
                symbol_lines.append(f"\n{rel}:")
                symbol_lines.extend(file_symbols)

        output.append("PYTHON SYMBOLS\n" + "─"*40 + "\n".join(symbol_lines))

        if imports_map:
            imp_str = "\n".join(f"  {f}: {', '.join(v[:6])}" for f, v in list(imports_map.items())[:15])
            output.append("IMPORT MAP (top 15 files)\n" + "─"*40 + "\n" + imp_str)

        if todos:
            output.append("TODOs / FIXMEs\n" + "─"*40 + "\n" + "\n".join(todos[:20]))
        else:
            output.append("TODOs / FIXMEs\n" + "─"*40 + "\nNone found.")

        return "\n\n".join(output)

    except PermissionError as e:
        return f"Security Error: {e}"
    except Exception as e:
        return f"Error indexing repo: {repr(e)}"


@tool
async def run_shell_command(command: str, timeout_seconds: int = 60) -> str:
    """Execute a shell command and return stdout + stderr.
    USE ONLY IN AUTONOMOUS MODE.
    For running tests prefer run_tests. For code execution prefer execute_code.
    Use for: pip install, npm, git, make, cargo, docker, curl, etc.
    Timeout default: 60s (increase for slow builds)."""
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(_PROJECT_ROOT),
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            proc.kill()
            return f"Command timed out after {timeout_seconds}s: `{command}`"

        parts = []
        if stdout:
            parts.append(f"STDOUT:\n{stdout.decode(errors='replace').strip()}")
        if stderr:
            parts.append(f"STDERR:\n{stderr.decode(errors='replace').strip()}")
        if not parts:
            parts.append("(no output)")

        status = "OK" if proc.returncode == 0 else f"Exit code {proc.returncode}"
        return f"[{status}] `{command}`\n{'─'*60}\n" + "\n".join(parts)
    except Exception as e:
        return f"Error running command: {repr(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# RESEARCH AGENT TOOLS
# ══════════════════════════════════════════════════════════════════════════════


@tool
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for developer resources: Stack Overflow answers, GitHub repos,
    official docs, tutorials, and error explanations.
    Use specific technical queries (e.g. 'Python asyncio deadlock fix site:stackoverflow.com')."""
    try:
        def _search() -> list[dict]:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))
        results = await asyncio.to_thread(_search)
        if not results:
            return f"No results for: '{query}'"
        blocks = [
            f"**{r.get('title','Untitled')}**\nURL: {r.get('href','')}\n{r.get('body','').strip()}"
            for r in results
        ]
        return "\n\n---\n\n".join(blocks)
    except Exception as e:
        return f"Search error: {repr(e)}"


@tool
async def scrape_page(url: str) -> str:
    """Fetch and extract clean text from a web page (up to 200 lines).
    Use after web_search when a snippet isn't enough — full SO answers, READMEs, docs."""
    try:
        def _scrape() -> str:
            headers = {"User-Agent": "Mozilla/5.0 (AgentOS/2.0)"}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            lines = [l.strip() for l in soup.get_text(separator="\n").splitlines() if l.strip()]
            return "\n".join(lines[:200])
        return await asyncio.to_thread(_scrape)
    except Exception as e:
        return f"Failed to scrape {url}: {repr(e)}"


@tool
async def search_algorithms(topic: str) -> str:
    """Search for algorithm implementations with time/space complexity analysis.
    Best for: sorting, graph traversal, dynamic programming, data structures, ML algorithms."""
    try:
        def _search() -> list[dict]:
            q = f"{topic} algorithm implementation time complexity Python"
            with DDGS() as ddgs:
                return list(ddgs.text(q, max_results=5))
        results = await asyncio.to_thread(_search)
        if not results:
            return f"No results for: '{topic}'"
        blocks = [
            f"**{r.get('title','Untitled')}**\nURL: {r.get('href','')}\n{r.get('body','').strip()}"
            for r in results
        ]
        return "\n\n---\n\n".join(blocks)
    except Exception as e:
        return f"Algorithm search error: {repr(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# DB AGENT TOOLS
# ══════════════════════════════════════════════════════════════════════════════


async def _get_user_id(config: Optional[RunnableConfig]) -> int:
    """Helper to extract active user_id from config, falling back to local dev user."""
    user_id = None
    if config and isinstance(config, dict) and "configurable" in config:
        user_id = config["configurable"].get("user_id")
    elif config and hasattr(config, "configurable") and config.configurable:
        user_id = config.configurable.get("user_id")
        
    if user_id:
        return int(user_id)
        
    # Lazy creation of local_dev_user for offline development
    from database import User
    import hashlib
    import os
    async with AsyncSessionLocal() as db:
        try:
            stmt = select(User).where(User.username == "local_dev_user")
            result = await db.execute(stmt)
            user = result.scalar_one_or_none()
            if not user:
                salt = os.urandom(16)
                key = hashlib.pbkdf2_hmac('sha256', b"local_dev_password", salt, 100000)
                hashed = salt.hex() + "$" + key.hex()
                user = User(username="local_dev_user", hashed_password=hashed)
                db.add(user)
                await db.commit()
                await db.refresh(user)
            return user.id
        except Exception as e:
            log.error("Failed to fetch/create local_dev_user: %s", e)
            # Wrote a fallback hardcoded dummy ID if DB is not ready
            return 1

@tool
async def save_code_snippet(
    title: str,
    code: str,
    language: str = "python",
    tags: Optional[str] = None,
    notes: Optional[str] = None,
    config: Optional[RunnableConfig] = None,
) -> str:
    """Save a code snippet to the database for future reuse.
    tags: comma-separated keywords  notes: what it does and when to use it."""
    user_id = await _get_user_id(config)
    async with AsyncSessionLocal() as db:
        try:
            db.add(CodeSnippet(user_id=user_id, title=title, code=code, language=language, tags=tags, notes=notes))
            await db.commit()
            return f"Saved snippet: '{title}' [{language}]" + (f" | Tags: {tags}" if tags else "")
        except Exception as e:
            await db.rollback()
            return f"DB Error: {e}"


@tool
async def get_snippets(
    query: Optional[str] = None, 
    language: Optional[str] = None,
    config: Optional[RunnableConfig] = None,
) -> str:
    """Retrieve saved code snippets filtered by keyword and/or language (up to 5)."""
    user_id = await _get_user_id(config)
    async with AsyncSessionLocal() as db:
        try:
            result   = await db.execute(
                select(CodeSnippet)
                .where(CodeSnippet.user_id == user_id)
                .order_by(CodeSnippet.created_at.desc())
            )
            snippets = result.scalars().all()
            if query:
                q        = query.lower()
                snippets = [s for s in snippets if q in s.title.lower() or (s.tags and q in s.tags.lower())]
            if language:
                snippets = [s for s in snippets if s.language.lower() == language.lower()]
            if not snippets:
                return "No matching snippets found."
            blocks = []
            for s in snippets[:5]:
                preview = s.code[:500] + ("..." if len(s.code) > 500 else "")
                blocks.append(f"### {s.title} [{s.language}]\nTags: {s.tags or 'none'}\n```{s.language}\n{preview}\n```")
            return "\n\n---\n\n".join(blocks)
        except Exception as e:
            return f"DB Error: {e}"


@tool
async def log_improvement(
    description: str, 
    category: str = "general", 
    impact: str = "medium",
    config: Optional[RunnableConfig] = None,
) -> str:
    """Log a code improvement, refactoring decision, or bug fix to the project changelog.
    category: bug-fix | refactor | performance | security | feature | general
    impact:   low | medium | high"""
    user_id = await _get_user_id(config)
    async with AsyncSessionLocal() as db:
        try:
            db.add(Improvement(user_id=user_id, description=description, category=category, impact=impact))
            await db.commit()
            return f"Logged [{category} | {impact}]: {description}"
        except Exception as e:
            await db.rollback()
            return f"DB Error: {e}"
