from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

LOGGER = logging.getLogger("llm_client")


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...


class MockClient(LLMClient):
    def generate(self, prompt: str) -> str:
        LOGGER.info("MockClient ativo – retornando resposta vazia.")
        return ""


class OpenAIClient(LLMClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai não instalado. Execute `pip install openai`") from exc

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY não configurado.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        LOGGER.info("OpenAIClient inicializado (%s)", model)

    def generate(self, prompt: str) -> str:
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Você gera datasets JSONL válidos. Retorne apenas JSON válidos por linha "
                            "sem explicações adicionais."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
            }
            if self.max_tokens:
                payload["max_tokens"] = self.max_tokens

            response = self.client.chat.completions.create(**payload)
            return response.choices[0].message.content or ""
        except Exception as exc:
            LOGGER.error("Erro na chamada OpenAI: %s", exc, exc_info=True)
            return ""


class GeminiClient(LLMClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash-lite",
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
    ) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError("google-generativeai não instalado. `pip install google-generativeai`") from exc

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY não configurado.")

        genai.configure(api_key=self.api_key)
        self._genai = genai
        self.model_name = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.model = genai.GenerativeModel(model_name=self.model_name)

        LOGGER.info("GeminiClient inicializado (%s)", model)

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            LOGGER.warning("Prompt vazio enviado ao Gemini.")
            return ""

        generation_config = {
            "temperature": self.temperature,
        }
        if self.max_output_tokens is not None:
            generation_config["max_output_tokens"] = self.max_output_tokens

        try:  # tentativa com modelo já carregado
            response = self.model.generate_content(
                [{"role": "user", "parts": [{"text": prompt}]}],
                generation_config=generation_config,
            )
            return getattr(response, "text", "") or ""
        except Exception as exc:
            LOGGER.error("Erro ao chamar Gemini (%s). Tentando fallback fixando modelo.", exc, exc_info=True)
            # tenta recarregar
            self.model = self._genai.GenerativeModel(model_name=self.model_name)
            try:
                response = self.model.generate_content(
                    [{"role": "user", "parts": [{"text": prompt}]}],
                    generation_config=generation_config,
                )
                return getattr(response, "text", "") or ""
            except Exception as final_exc:
                LOGGER.error("Falha final com Gemini: %s", final_exc, exc_info=True)
                return ""


CLIENT_MAP = {
    "mock": MockClient,
    "openai": OpenAIClient,
    "gemini": GeminiClient,
}


def get_client(name: str, **kwargs) -> LLMClient:
    if name not in CLIENT_MAP:
        raise ValueError(f"Cliente '{name}' não suportado. Opções: {list(CLIENT_MAP)}")
    return CLIENT_MAP[name](**kwargs)
