from __future__ import annotations

import logging
import os
import re
import time
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

        self._genai = genai
        self.model_name = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.api_keys = self._resolve_api_keys(api_key)
        self._current_key_index = 0
        self.requests_per_minute = int(os.getenv("GEMINI_REQUESTS_PER_MINUTE", "15"))
        self._min_interval = 60.0 / self.requests_per_minute if self.requests_per_minute > 0 else 0.0
        self._key_last_call = [0.0 for _ in self.api_keys]
        self._max_attempts_per_key = max(1, int(os.getenv("GEMINI_MAX_ATTEMPTS_PER_KEY", "2")))
        self._configure_client()

        LOGGER.info(
            "GeminiClient inicializado (%s) com %d chave(s) configuradas",
            model,
            len(self.api_keys),
        )

    @staticmethod
    def _resolve_api_keys(explicit_key: Optional[str]) -> list[str]:
        def _append_if_present(container: list[str], value: Optional[str]) -> None:
            if value and value not in container:
                container.append(value)

        keys: list[str] = []
        _append_if_present(keys, explicit_key)

        # suporta GEMINI_API_KEY, GEMINI_API_KEY_1..3 e GEMINI_API_KEYS (lista separada por vírgulas)
        _append_if_present(keys, os.getenv("GEMINI_API_KEY"))
        for idx in range(1, 4):
            _append_if_present(keys, os.getenv(f"GEMINI_API_KEY_{idx}"))

        bulk_keys = os.getenv("GEMINI_API_KEYS")
        if bulk_keys:
            for value in bulk_keys.split(","):
                _append_if_present(keys, value.strip())

        if not keys:
            raise ValueError("Nenhuma chave configurada para o Gemini. Defina GEMINI_API_KEY ou variantes numeradas.")
        return keys

    def _configure_client(self) -> None:
        current_key = self.api_keys[self._current_key_index]
        self._genai.configure(api_key=current_key)
        self.model = self._genai.GenerativeModel(model_name=self.model_name)

    def _respect_rate_limit(self) -> None:
        if self._min_interval <= 0:
            return
        idx = self._current_key_index
        last_call = self._key_last_call[idx]
        elapsed = time.monotonic() - last_call
        wait_time = self._min_interval - elapsed
        if wait_time > 0:
            LOGGER.debug(
                "Aguardando %.2fs para respeitar rate limit da chave #%d",
                wait_time,
                idx + 1,
            )
            time.sleep(wait_time)

    def _mark_call_complete(self) -> None:
        self._key_last_call[self._current_key_index] = time.monotonic()

    def _rotate_key(self) -> bool:
        if self._current_key_index + 1 >= len(self.api_keys):
            return False
        self._current_key_index += 1
        LOGGER.warning("Alternando para GEMINI_API_KEY #%d", self._current_key_index + 1)
        self._configure_client()
        return True

    def _advance_key_after_success(self) -> None:
        if len(self.api_keys) <= 1:
            return
        self._current_key_index = (self._current_key_index + 1) % len(self.api_keys)
        self._configure_client()

    def _current_key_label(self) -> str:
        suffix = self.api_keys[self._current_key_index]
        return f"...{suffix[-4:]}" if len(suffix) >= 4 else "***"

    @staticmethod
    def _extract_retry_delay(exc: Exception) -> Optional[float]:
        message = str(exc)
        match = re.search(r"retry in (\d+(?:\.\d+)?)s", message, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            LOGGER.warning("Prompt vazio enviado ao Gemini.")
            return ""

        generation_config = {
            "temperature": self.temperature,
        }
        if self.max_output_tokens is not None:
            generation_config["max_output_tokens"] = self.max_output_tokens

        def _call_model() -> str:
            response = self.model.generate_content(
                [{"role": "user", "parts": [{"text": prompt}]}],
                generation_config=generation_config,
            )
            return getattr(response, "text", "") or ""

        attempts_with_keys = 0
        attempts_current_key = 0
        while attempts_with_keys < len(self.api_keys):
            try:
                self._respect_rate_limit()
                result = _call_model()
                self._mark_call_complete()
                self._advance_key_after_success()
                return result
            except Exception as exc:
                LOGGER.error(
                    "Erro ao chamar Gemini com chave #%d (%s): %s.",
                    self._current_key_index + 1,
                    self._current_key_label(),
                    exc,
                    exc_info=True,
                )
                retry_delay = self._extract_retry_delay(exc)
                if retry_delay:
                    LOGGER.warning("⏱️  Rate limit atingido; aguardando %.2fs antes de tentar novamente.", retry_delay)
                    time.sleep(min(retry_delay, 120))
                try:
                    self._configure_client()
                except Exception as reload_exc:
                    LOGGER.error(
                        "Falha ao reconfigurar chave #%d: %s",
                        self._current_key_index + 1,
                        reload_exc,
                        exc_info=True,
                    )
                    attempts_current_key = self._max_attempts_per_key
                else:
                    attempts_current_key += 1
                    if attempts_current_key < self._max_attempts_per_key:
                        continue

                attempts_with_keys += 1
                attempts_current_key = 0
                if not self._rotate_key():
                    break
                LOGGER.info(
                    "Repetindo tentativa com chave #%d (%s).",
                    self._current_key_index + 1,
                    self._current_key_label(),
                )

        LOGGER.error("Todas as chaves do Gemini falharam para a requisição atual.")
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
