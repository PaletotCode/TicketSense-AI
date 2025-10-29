"""
Interfaces e clientes para APIs de LLM.
Arquitetura baseada em ABC para facilitar a adição de novos provedores.
"""
from __future__ import annotations
import os
import logging
from abc import ABC, abstractmethod
from typing import Optional

# Configura o logging
LOGGER = logging.getLogger("llm_client")


class LLMClient(ABC):
    """Interface abstrata para um cliente de geração de texto."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Envia um prompt e retorna a resposta de texto bruto.
        
        Args:
            prompt: O prompt a ser enviado ao LLM
            
        Returns:
            String contendo a resposta do LLM
        """
        pass


class MockClient(LLMClient):
    """Cliente falso para testes, não usa API real."""
    
    def generate(self, prompt: str) -> str:
        """Retorna dados de teste sem chamar API externa."""
        LOGGER.info("Usando MockClient. Retornando 2 amostras falsas.")
        return """{"messages": [{"role": "user", "text": "Texto falso de teste 1.", "intent": ["MOCK"]}]}
{"messages": [{"role": "user", "text": "Texto falso de teste 2.", "intent": ["MOCK"]}]}"""


class OpenAIClient(LLMClient):
    """Cliente real para a API da OpenAI."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ):
        """
        Inicializa o cliente OpenAI.
        
        Args:
            api_key: Chave da API (se None, busca do ambiente)
            model: Modelo a ser usado (default: gpt-3.5-turbo)
            temperature: Controle de aleatoriedade (0.0 a 2.0)
            max_tokens: Limite máximo de tokens na resposta
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI não está instalado. Por favor, execute: pip install openai"
            )
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Chave da API OpenAI não encontrada. "
                "Defina OPENAI_API_KEY no seu .env ou passe como parâmetro."
            )
            
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        LOGGER.info(
            "Cliente OpenAI inicializado para o modelo %s (temp=%.2f)", 
            self.model, 
            self.temperature
        )

    def generate(self, prompt: str) -> str:
        """Gera texto usando a API da OpenAI."""
        try:
            kwargs = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system", 
                        "content": "Você é um assistente especializado em gerar dados JSONL de alta qualidade para treinamento de modelos de IA. Sempre retorne apenas JSONL válido, sem texto adicional."
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
            }
            
            if self.max_tokens:
                kwargs["max_tokens"] = self.max_tokens
            
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            
            LOGGER.debug(
                "Resposta da OpenAI recebida: %d caracteres, %d tokens usados",
                len(content) if content else 0,
                response.usage.total_tokens if response.usage else 0
            )
            
            return content if content else ""
            
        except Exception as e:
            LOGGER.error("Erro na API OpenAI: %s", e, exc_info=True)
            return ""  # Retorna string vazia em caso de falha


class GeminiClient(LLMClient):
    """Cliente para a API do Google Gemini."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-pro",
        temperature: float = 0.7
    ):
        """
        Inicializa o cliente Gemini.
        
        Args:
            api_key: Chave da API (se None, busca do ambiente)
            model: Modelo a ser usado (default: gemini-pro)
            temperature: Controle de aleatoriedade (0.0 a 1.0)
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "Google Generative AI não está instalado. "
                "Por favor, execute: pip install google-generativeai"
            )
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Chave da API Gemini não encontrada. "
                "Defina GEMINI_API_KEY no seu .env ou passe como parâmetro."
            )
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
        self.temperature = temperature
        
        LOGGER.info(
            "Cliente Gemini inicializado para o modelo %s (temp=%.2f)",
            model,
            self.temperature
        )
    
    def generate(self, prompt: str) -> str:
        """Gera texto usando a API do Gemini."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                }
            )
            
            content = response.text
            LOGGER.debug(
                "Resposta do Gemini recebida: %d caracteres",
                len(content) if content else 0
            )
            
            return content if content else ""
            
        except Exception as e:
            LOGGER.error("Erro na API Gemini: %s", e, exc_info=True)
            return ""


# Mapeamento para fácil seleção via CLI
CLIENT_MAP = {
    "mock": MockClient,
    "openai": OpenAIClient,
    "gemini": GeminiClient,
}


def get_client(client_name: str, **kwargs) -> LLMClient:
    """
    Factory function para criar clientes LLM.
    
    Args:
        client_name: Nome do cliente ("mock", "openai", "gemini")
        **kwargs: Argumentos adicionais para o construtor do cliente
        
    Returns:
        Instância do cliente LLM solicitado
        
    Raises:
        ValueError: Se o nome do cliente for inválido
    """
    if client_name not in CLIENT_MAP:
        raise ValueError(
            f"Cliente '{client_name}' não encontrado. "
            f"Opções disponíveis: {list(CLIENT_MAP.keys())}"
        )
    
    client_class = CLIENT_MAP[client_name]
    return client_class(**kwargs)