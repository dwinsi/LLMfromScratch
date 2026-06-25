"""
Tool definitions for the coding agent.

Seven tools covering the core operations of a coding agent:
  read_file       - read any file's contents
  write_file      - write or overwrite a file
  list_directory  - list files in a folder
  run_bash        - execute a shell command
  search_in_file  - find text in a file
  save_memory     - persist a fact to global or project memory
  recall_memory   - retrieve all saved facts

Each tool has a clear docstring. LangChain uses the docstring
as the tool description shown to the model, so it must be precise.
The model reads these descriptions to decide which tool to call.
"""

import os
import subprocess
from langchain_core.tools import tool

from memory_store import MemoryStore

# Injected at startup by cli.py once the project directory is known.
_memory_store: MemoryStore | None = None


def set_memory_store(store: MemoryStore) -> None:
    global _memory_store
    _memory_store = store


@tool
def read_file(file_path: str) -> str:
    """
    Read the full contents of a file and return them as a string.
    Use this when you need to understand what a file contains before modifying it.
    Input: the path to the file, relative or absolute.
    Returns the file contents as a string, or an error message if the file does not exist.
    """
    try:
        file_path = os.path.expanduser(file_path)
        if not os.path.exists(file_path):
            return f"Error: file not found at path '{file_path}'"
        if os.path.getsize(file_path) > 100_000:
            return f"Error: file is too large to read in full (>{100_000} bytes). Use search_in_file instead."
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            contents = f.read()
        return contents if contents else "(file is empty)"
    except Exception as error:
        return f"Error reading file: {error}"


@tool
def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file. Creates the file if it does not exist.
    Overwrites the file if it already exists, so use read_file first
    if you want to preserve existing content.
    Input: file_path (string), content (string to write).
    Returns a confirmation message or an error.
    """
    try:
        file_path = os.path.expanduser(file_path)

        # Create parent directories if they do not exist
        parent_directory = os.path.dirname(file_path)
        if parent_directory:
            os.makedirs(parent_directory, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return f"Successfully wrote {len(content)} characters to '{file_path}'"
    except Exception as error:
        return f"Error writing file: {error}"


@tool
def list_directory(directory_path: str = ".") -> str:
    """
    List all files and folders in a directory.
    Use this to understand the structure of a codebase before reading individual files.
    Input: directory path (defaults to current directory if not provided).
    Returns a formatted list of files and folders with their sizes.
    """
    try:
        directory_path = os.path.expanduser(directory_path)
        if not os.path.exists(directory_path):
            return f"Error: directory not found at path '{directory_path}'"
        if not os.path.isdir(directory_path):
            return f"Error: '{directory_path}' is a file, not a directory"

        entries = []
        for entry in sorted(os.listdir(directory_path)):
            full_path   = os.path.join(directory_path, entry)
            entry_type  = "DIR " if os.path.isdir(full_path) else "FILE"
            if os.path.isfile(full_path):
                size_bytes = os.path.getsize(full_path)
                size_label = f"{size_bytes:>8,} bytes"
            else:
                size_label = "         -    "
            entries.append(f"  {entry_type}  {size_label}  {entry}")

        if not entries:
            return f"Directory '{directory_path}' is empty."

        header = f"Contents of '{directory_path}':\n"
        return header + "\n".join(entries)

    except Exception as error:
        return f"Error listing directory: {error}"


@tool
def run_bash(command: str) -> str:
    """
    Execute a bash shell command and return its output.
    Use this to run Python scripts, tests, git commands, or any shell operation.
    Input: the shell command as a string.
    Returns the combined stdout and stderr output, or an error message.

    Safety: avoid commands that could cause irreversible damage such as
    rm -rf, format, or anything that modifies system files.
    Always prefer targeted commands over broad destructive ones.
    """
    # Basic safety check for obviously dangerous commands
    dangerous_patterns = ['rm -rf /', 'format c:', 'mkfs', ':(){:|:&};:']
    command_lower = command.lower()
    for pattern in dangerous_patterns:
        if pattern in command_lower:
            return f"Error: command blocked for safety reasons. Do not use destructive system commands."

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,          # 30 second timeout prevents hanging
            encoding='utf-8',
            errors='replace',
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if not output:
            output = "(command completed with no output)"

        # Include return code if non-zero
        if result.returncode != 0:
            output += f"\n(exit code: {result.returncode})"

        return output.strip()

    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds."
    except Exception as error:
        return f"Error running command: {error}"


@tool
def search_in_file(file_path: str, search_query: str) -> str:
    """
    Search for a string or pattern in a file and return matching lines with line numbers.
    Use this to find function definitions, variable usages, error messages, or specific code patterns
    without reading the entire file.
    Input: file_path (string), search_query (string to search for, case-insensitive).
    Returns all matching lines with their line numbers, or a message if nothing is found.
    """
    try:
        file_path = os.path.expanduser(file_path)
        if not os.path.exists(file_path):
            return f"Error: file not found at path '{file_path}'"

        matching_lines = []
        search_lower   = search_query.lower()

        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line_number, line in enumerate(f, start=1):
                if search_lower in line.lower():
                    matching_lines.append(f"  Line {line_number:4d}: {line.rstrip()}")

        if not matching_lines:
            return f"No matches found for '{search_query}' in '{file_path}'"

        header = f"Found {len(matching_lines)} match(es) for '{search_query}' in '{file_path}':\n"
        return header + "\n".join(matching_lines)

    except Exception as error:
        return f"Error searching file: {error}"


@tool
def save_memory(scope: str, key: str, value: str) -> str:
    """
    Save an important fact to persistent memory so it is remembered in future sessions.
    Use this whenever you learn something worth keeping — project structure, user preferences,
    architectural decisions, recurring patterns, or anything the user explicitly asks you to remember.

    scope: 'global' for user preferences that apply across all projects,
           'project' for facts specific to the current codebase.
    key:   a short descriptive label, e.g. 'entry_point' or 'preferred_style'.
    value: the fact to remember, e.g. 'cli.py' or 'concise answers, no preamble'.

    Examples:
      save_memory('project', 'entry_point', 'cli.py')
      save_memory('global', 'preferred_language', 'Python')
      save_memory('project', 'llm_framework', 'langchain 1.x with create_agent')
    """
    if _memory_store is None:
        return "Memory is not available in this session."
    return _memory_store.save(scope, key, value)


@tool
def recall_memory() -> str:
    """
    Retrieve all facts saved in persistent memory (both global and project-level).
    Use this at the start of a task when you need context about the project or user preferences,
    or when the user asks what you remember.
    Returns a formatted list of all saved facts, separated by scope.
    """
    if _memory_store is None:
        return "Memory is not available in this session."
    all_facts = _memory_store.recall_all()
    if not all_facts:
        return "No facts saved in memory yet."
    lines = []
    global_facts  = _memory_store.recall_global()
    project_facts = _memory_store.recall_project()
    if global_facts:
        lines.append("Global memory:")
        for k, v in global_facts.items():
            lines.append(f"  {k}: {v}")
    if project_facts:
        lines.append("Project memory:")
        for k, v in project_facts.items():
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def get_all_tools() -> list:
    """Return all tools as a list for the agent."""
    return [
        read_file,
        write_file,
        list_directory,
        run_bash,
        search_in_file,
        save_memory,
        recall_memory,
    ]