""""""
from __future__ import annotations
import json
from typing import Iterable, List, Dict, Any


def build_minimalist_prompt(intents: List[str], batch_size: int) -> str:
    """"""
    intent_str = json.dumps(intents, ensure_ascii=False)
    json_format = f'{{"messages": [{{"role": "user", "text": "TEXTO_GERADO_AQUI", "intent": {intent_str}}}]}}'
    
    prompt = f""""""
    
    return prompt.strip()


def iter_json_objects(raw_text: str) -> Iterable[Dict[str, Any]]:
    """"""
    decoder = json.JSONDecoder()
    idx = 0
    length = len(raw_text)

    while idx < length:
        raw_text = raw_text.lstrip()
        if not raw_text:
            break
            
        try:
            obj, offset = decoder.raw_decode(raw_text)
            yield obj
            raw_text = raw_text[offset:]
            idx += offset
        except json.JSONDecodeError:
            raw_text = raw_text[1:]
            idx += 1


def validate_sample(sample: Dict[str, Any], expected_intents: List[str]) -> bool:
    """"""
    try:
        if "messages" not in sample or not isinstance(sample["messages"], list):
            return False
        
        if len(sample["messages"]) == 0:
            return False
        
        message = sample["messages"][0]
        if "role" not in message or "text" not in message or "intent" not in message:
            return False
        if not message["text"] or len(message["text"].strip()) == 0:
            return False
        if set(message["intent"]) != set(expected_intents):
            return False
        
        return True
    except (KeyError, TypeError, IndexError):
        return False