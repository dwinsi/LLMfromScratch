from saathi.tools.filesystem import list_directory, patch_file, read_file, write_file
from saathi.tools.git import (
    git_commit,
    git_diff,
    git_diff_staged,
    git_log,
    git_status,
)
from saathi.tools.memory_tools import recall_memory, save_memory
from saathi.tools.search import search_across_files, search_in_file, search_web
from saathi.tools.shell import run_bash

ALL_TOOLS = [
    read_file,
    write_file,
    patch_file,
    list_directory,
    run_bash,
    search_in_file,
    search_across_files,
    search_web,
    save_memory,
    recall_memory,
    git_status,
    git_diff,
    git_diff_staged,
    git_log,
    git_commit,
]

__all__ = ["ALL_TOOLS"]
