from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List


def build_minimalist_prompt(intents: List[str], batch_size: int) -> str:
    intent_str = json.dumps(intents, ensure_ascii=False)
    json_format = '{"messages": [{"role": "user", "text": "TEXTO_GERADO_AQUI", "intent": ' + intent_str + '}]}'

    lines = [
        f"Gere {batch_size} exemplos de texto de usuário, em português do Brasil, para um serviço de atendimento ao cliente.",
        f"A intenção de TODOS os {batch_size} exemplos deve ser exatamente: {intent_str}.",
        "",
        "Requisitos importantes:",
        "- Textos realistas e variados (erros de digitação, gírias, diferentes níveis de formalidade)",
        "- Frases curtas e longas",
        "- Diferentes formas de expressar a mesma intenção",
        "",
        "Responda usando APENAS o seguinte formato JSONL (um JSON válido por linha, sem numeração ou texto adicional):",
        json_format,
        json_format,
        "...",
    ]

    prompt = "\n".join(lines)
    return prompt.strip()


def iter_json_objects(raw_text: str) -> Iterable[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    idx = 0
    buffer = raw_text

    while idx < len(buffer):
        buffer = buffer.lstrip()
        if not buffer:
            break
        try:
            obj, offset = decoder.raw_decode(buffer)
            yield obj
            buffer = buffer[offset:]
            idx += offset
        except json.JSONDecodeError:
            buffer = buffer[1:]
            idx += 1


def validate_sample(sample: Dict[str, Any], expected_intents: List[str]) -> bool:
    try:
        if "messages" not in sample or not isinstance(sample["messages"], list):
            return False
        if len(sample["messages"]) == 0:
            return False
        message = sample["messages"][0]
        if not all(key in message for key in ("role", "text", "intent")):
            return False
        if not message["text"] or len(message["text"].strip()) == 0:
            return False
        if set(message["intent"]) != set(expected_intents):
            return False
        return True
    except (KeyError, TypeError, IndexError):
        return False
