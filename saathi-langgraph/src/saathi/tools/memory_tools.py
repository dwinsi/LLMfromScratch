"""Memory read/write tools that delegate to MemoryStore."""

from langchain_core.tools import tool

from saathi.memory.store import MemoryStore

_store = MemoryStore()


@tool
def save_memory(scope: str, key: str, value: str) -> str:
    """
    Persist a fact for future sessions.
    scope: 'global' (user-level, ~/.saathi/) or 'project' (.saathi/ in cwd).
    Example: save_memory('project', 'entry_point', 'src/main.py')
    """
    if scope not in ("global", "project"):
        return "Error: scope must be 'global' or 'project'"
    _store.save(scope, key, value)
    return f"Saved [{scope}] {key} = {value}"


@tool
def recall_memory(scope: str = "all") -> str:
    """
    Retrieve saved facts.
    scope: 'global', 'project', or 'all' (default).
    """
    data = _store.all()
    if scope == "global":
        facts = data["global"]
    elif scope == "project":
        facts = data["project"]
    else:
        facts = {**data["global"], **data["project"]}

    if not facts:
        return "No facts saved yet."
    lines = [f"  {k}: {v}" for k, v in facts.items()]
    return "\n".join(lines)
