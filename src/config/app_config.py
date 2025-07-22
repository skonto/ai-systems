from pydantic_settings import BaseSettings, SettingsConfigDict

from rag import ModelConfig


class Settings(BaseSettings):
    model: ModelConfig = ModelConfig()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="QA_APP_",
    )

def get_settings() -> Settings:
    return Settings()
