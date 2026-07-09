"""LangGraph node implementations."""

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from saathi.agent.prompts import build_system_prompt
from saathi.agent.state import AgentState
from saathi.config import settings
from saathi.logging_config import get_logger
from saathi.memory.store import MemoryStore
from saathi.project_context import find_project_instructions
from saathi.retry import retry_async

log = get_logger()


def make_agent_node(llm: LanguageModelLike, memory_store: MemoryStore):
    """Return a LangGraph node that calls the LLM with bound tools.

    Accepts a ``LanguageModelLike`` because ``ChatOllama.bind_tools(...)`` returns
    a ``Runnable`` binding rather than a bare ``ChatOllama``.
    """

    # Loaded once per graph build; a new session picks up edited SAATHI.md.
    project_instructions = find_project_instructions()

    def _on_retry(attempt: int, exc: BaseException, delay: float) -> None:
        log.warning(
            "ollama_retry",
            attempt=attempt,
            delay=delay,
            error=str(exc) or type(exc).__name__,
        )

    async def agent_node(state: AgentState, config: RunnableConfig) -> dict:
        memory_block = memory_store.format_for_prompt()
        system_prompt = build_system_prompt(
            context_paths=state.get("context_paths", []),
            memory_block=memory_block,
            mode=state.get("mode", "default"),
            project_instructions=project_instructions,
        )

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        # Retry only transient connection failures (server not up yet); a slow
        # response is not retried — see saathi.retry.
        response = await retry_async(
            lambda: llm.ainvoke(messages, config),
            attempts=settings.ollama_max_retries,
            base_delay=settings.ollama_retry_base_delay,
            on_retry=_on_retry,
        )
        return {"messages": [response]}

    return agent_node
