"""CLI entry point: `saathi-api` starts the uvicorn server."""

import uvicorn


def main():
    uvicorn.run(
        "saathi.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
