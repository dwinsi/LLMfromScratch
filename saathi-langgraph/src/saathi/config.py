from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SAATHI_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_model: str = "gemma4:12b"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    context_window: int = 32768
    max_tokens: int = 4096
    max_parallel_tools: int = 8
    ollama_max_retries: int = 3
    ollama_retry_base_delay: float = 1.0
    review_min_confidence: int = 70
    brave_api_key: str | None = None
    debug: bool = False

    @property
    def history_token_budget(self) -> int:
        return int(self.context_window * 0.75)


settings = Settings()
