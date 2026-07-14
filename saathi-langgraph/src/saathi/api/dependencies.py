"""Shared singletons injected into route handlers via FastAPI Depends."""

from typing import Annotated

from fastapi import Depends

# Populated during lifespan startup; routes import get_graph / get_memory_store.
_graph = None
_memory_store = None


def get_graph():
    if _graph is None:
        raise RuntimeError("Agent graph not initialised — server still starting up.")
    return _graph


def get_memory_store():
    if _memory_store is None:
        raise RuntimeError("Memory store not initialised — server still starting up.")
    return _memory_store


GraphDep = Annotated[object, Depends(get_graph)]
MemoryDep = Annotated[object, Depends(get_memory_store)]
