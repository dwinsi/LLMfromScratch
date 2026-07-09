"""A minimal MCP server over stdio — an example (and test fixture) for Saathi.

Run it via Saathi by copying examples/mcp.example.json to .saathi/mcp.json.
It exposes a single `echo` tool. Requires the `mcp` package (a Saathi dependency).
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo")


@mcp.tool()
def echo(text: str) -> str:
    """Echo the given text back, prefixed with 'echo: '."""
    return f"echo: {text}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
