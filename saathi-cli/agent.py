"""
saathi-cli
===========
A coding companion built with Gemma 4 via Ollama and LangChain ReAct.

Saathi means companion in Hindi. This tool walks alongside you in your codebase.
It reads, writes, navigates and runs code on your behalf.

Structure:
  agent.py          <- you are here, the ReAct loop
  tools.py          <- five tool definitions
  system_prompt.py  <- agent identity and behaviour
  cli.py            <- terminal interface

Install:
  pip install -r requirements.txt

Usage:
  python agent.py         <- single test task
  python cli.py           <- interactive terminal session
"""

from langchain.agents import create_agent
from langchain_core.messages import trim_messages
from langchain_ollama import ChatOllama

from tools import get_all_tools
from system_prompt import build_system_prompt


# ---- Configuration ----

OLLAMA_MODEL    = "gemma4:12b"   # change to gemma4:27b if you have the VRAM
OLLAMA_BASE_URL = "http://localhost:11434"
TEMPERATURE     = 0.1            # low temperature for consistent tool calling

# Context window: Ollama defaults to 2048 which causes trimmed responses.
# Gemma 4 supports up to 128k. Raise if you have the RAM.
# num_ctx:     total tokens the model can see (prompt + history + output)
# num_predict: max tokens the model can generate in a single response (-1 = unlimited)
CTX_WINDOW  = 32768   # 32k — safe default; bump to 65536 or 131072 if needed
MAX_PREDICT = 4096    # plenty for code explanations; raise for very long outputs


def load_llm() -> ChatOllama:
    """
    Connect to Gemma 4 running in Ollama.
    Ollama must be running before calling this.
    The model is served at localhost:11434 by default.
    """
    print(f"Connecting to Ollama model: {OLLAMA_MODEL}")
    print(f"Make sure Ollama is running: ollama serve")
    print(f"Make sure model is pulled:   ollama pull {OLLAMA_MODEL}\n")

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
        num_ctx=CTX_WINDOW,
        num_predict=MAX_PREDICT,
    )
    return llm


def build_agent(
    llm: ChatOllama,
    context_paths: list[str] | None = None,
    memory_block: str = "",
):
    """
    Build the LangChain agent with all tools.

    langchain 1.x uses create_agent (replaces create_react_agent + AgentExecutor).
    The agent runs a tool-calling loop internally until it reaches a final answer.
    context_paths: optional list of files/folders to scope the agent's attention to.
    memory_block:  pre-formatted memory facts injected into the system prompt.
    """
    tools = get_all_tools()

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=build_system_prompt(context_paths, memory_block),
    )

    return agent


# Reserve 25% of the window for the model's response.
# trim_messages drops the oldest messages (keeping the system message) to stay under budget.
HISTORY_TOKEN_BUDGET = int(CTX_WINDOW * 0.75)


def compact_history(messages: list) -> list:
    """Trim conversation history to fit within the token budget."""
    if not messages:
        return messages
    return trim_messages(
        messages,
        max_tokens=HISTORY_TOKEN_BUDGET,
        token_counter="approximate",
        strategy="last",       # drop oldest messages first
        include_system=True,   # always keep the system prompt
        start_on="human",      # trimmed history must start with a human message
    )


def run_task(agent, task: str) -> str:
    """Run a single task through the agent and return the final answer."""
    result = agent.invoke({"messages": [("human", task)]})
    messages = result.get("messages", [])
    if messages:
        return messages[-1].content
    return "No output returned."


# ---- Quick test when run directly ----

if __name__ == "__main__":
    llm            = load_llm()
    agent_executor = build_agent(llm)

    test_task = "List the files in the current directory and tell me what each one does."
    print(f"Task: {test_task}\n")
    print("-" * 60)

    answer = run_task(agent_executor, test_task)
    print("\nFinal answer:")
    print(answer)