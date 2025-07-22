from pydantic import BaseModel


class ModelConfig(BaseModel):
    name: str = "mistral"
    quantization: str = "bnb"
    max_tokens: int = 2048
