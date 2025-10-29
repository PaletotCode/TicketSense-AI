"""
Utilitários para o pipeline de geração de dataset.
"""
from __future__ import annotations
import json
from typing import Iterable, List, Dict, Any


def build_minimalist_prompt(intents: List[str], batch_size: int) -> str:
    """
    Cria o prompt focado para o LLM, solicitando dados no formato JSONL mínimo.
    
    Args:
        intents: Lista de intenções para gerar (ex: ["PAYMENT", "BILLING"])
        batch_size: Número de exemplos a solicitar
        
    Returns:
        String contendo o prompt formatado para o LLM
    """
    intent_str = json.dumps(intents, ensure_ascii=False)
    
    # O formato JSONL mínimo que nosso parser de treino (dataset_utils.py) entende
    json_format = f'{{"messages": [{{"role": "user", "text": "TEXTO_GERADO_AQUI", "intent": {intent_str}}}]}}'
    
    prompt = f"""Gere {batch_size} exemplos de texto de usuário, em português do Brasil, para um serviço de atendimento ao cliente.
A intenção de TODOS os {batch_size} exemplos deve ser exatamente: {intent_str}.

Requisitos importantes:
- Textos realistas e variados (erros de digitação, gírias, diferentes níveis de formalidade)
- Frases curtas e longas
- Diferentes formas de expressar a mesma intenção

Responda usando APENAS o seguinte formato JSONL (um JSON válido por linha, sem numeração ou texto adicional):
{json_format}
{json_format}
..."""
    
    return prompt.strip()


def iter_json_objects(raw_text: str) -> Iterable[Dict[str, Any]]:
    """
    Faz parsing incremental de objetos JSON concatenados ou mal formatados.
    Esta função é vital para limpar a saída do LLM que pode conter texto extra.
    
    Args:
        raw_text: Texto bruto retornado pelo LLM (pode conter múltiplos JSONs)
        
    Yields:
        Dicionários Python representando cada objeto JSON válido encontrado
    """
    decoder = json.JSONDecoder()
    idx = 0
    length = len(raw_text)

    while idx < length:
        # Remove espaços em branco iniciais
        raw_text = raw_text.lstrip()
        if not raw_text:
            break
            
        try:
            # Tenta decodificar um objeto JSON a partir da posição atual
            obj, offset = decoder.raw_decode(raw_text)
            yield obj
            # Avança para após o objeto decodificado
            raw_text = raw_text[offset:]
            idx += offset
        except json.JSONDecodeError:
            # Se falhar, pula um caractere e tenta novamente
            # (Isso lida com texto extra, como "Aqui estão os exemplos: {...}")
            raw_text = raw_text[1:]
            idx += 1


def validate_sample(sample: Dict[str, Any], expected_intents: List[str]) -> bool:
    """
    Valida se uma amostra gerada está no formato correto.
    
    Args:
        sample: Dicionário representando uma amostra
        expected_intents: Lista de intenções esperadas
        
    Returns:
        True se a amostra é válida, False caso contrário
    """
    try:
        # Verifica estrutura básica
        if "messages" not in sample or not isinstance(sample["messages"], list):
            return False
        
        if len(sample["messages"]) == 0:
            return False
        
        message = sample["messages"][0]
        
        # Verifica campos obrigatórios
        if "role" not in message or "text" not in message or "intent" not in message:
            return False
        
        # Verifica se o texto não está vazio
        if not message["text"] or len(message["text"].strip()) == 0:
            return False
        
        # Verifica se as intenções correspondem
        if set(message["intent"]) != set(expected_intents):
            return False
        
        return True
    except (KeyError, TypeError, IndexError):
        return False