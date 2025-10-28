"""
Utilitário para normalizar arquivos JSONL gerados por modelos.

Uso:
    python scripts/format_dataset.py \
        --input dataset.jsonl \
        --output dataset_formatted.jsonl

Se --output não for informado, o script sobrescreve o arquivo de entrada.

O script aceita arquivos onde diversos objetos JSON estejam concatenados
na mesma linha e reescreve cada objeto em uma linha separada, garantindo
um JSONL válido e formatado.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def iter_json_objects(raw_text: str) -> Iterable[dict]:
    """Faz parsing incremental de objetos JSON concatenados."""
    decoder = json.JSONDecoder()
    idx = 0
    length = len(raw_text)

    while idx < length:
        raw_text = raw_text.lstrip()
        if not raw_text:
            break
        obj, offset = decoder.raw_decode(raw_text)
        yield obj
        raw_text = raw_text[offset:]
        idx += offset


def normalize_jsonl(input_path: Path, output_path: Path) -> None:
    """Normaliza o arquivo JSONL garantindo um objeto por linha."""
    raw_text = input_path.read_text(encoding="utf-8")
    objects = list(iter_json_objects(raw_text))

    with output_path.open("w", encoding="utf-8") as fout:
        for obj in objects:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Formata JSONL gerado por IA.")
    parser.add_argument("--input", required=True, help="Arquivo JSONL original.")
    parser.add_argument(
        "--output",
        help="Arquivo de saída. Se omitido, sobrescreve o arquivo de entrada.",
    )

    args = parser.parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else input_path

    normalize_jsonl(input_path, output_path)
    print(f"Arquivo formatado salvo em {output_path}")


if __name__ == "__main__":
    main()
