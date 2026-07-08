"""Package imports and tool registration."""

import saathi

EXPECTED_TOOLS = {
    "read_file",
    "write_file",
    "patch_file",
    "list_directory",
    "run_bash",
    "search_in_file",
    "search_across_files",
    "search_web",
    "save_memory",
    "recall_memory",
    "git_status",
    "git_diff",
    "git_diff_staged",
    "git_log",
    "git_commit",
}


def test_version() -> None:
    assert saathi.__version__


def test_all_tools_registered() -> None:
    from saathi.tools import ALL_TOOLS

    names = {t.name for t in ALL_TOOLS}
    assert names == EXPECTED_TOOLS
    assert len(ALL_TOOLS) == len(EXPECTED_TOOLS)  # no duplicates


def test_core_modules_import() -> None:
    import saathi.agent.graph  # noqa: F401
    import saathi.agent.tool_node  # noqa: F401
    import saathi.cli  # noqa: F401
    import saathi.diagnostics  # noqa: F401
    import saathi.hooks.runner  # noqa: F401
    import saathi.project_context  # noqa: F401
