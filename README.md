# PingFy_IA

Pipeline modular de classificação de intenções para a plataforma PingFy.

## Estrutura

```
pingfy_ia/
├── api/               # API FastAPI de inferência
├── config/            # Configurações centralizadas
├── trainer/           # Utilitários e pipeline de treinamento
├── scripts/           # Ferramentas auxiliares (prep. dataset)
├── artifacts/         # Checkpoints, best_model e logs gerados
└── data/              # Dataset local (copiado ou baixado)
```

## Como treinar (Mac M4 com MPS)

```bash
source venv/bin/activate
python -m trainer.train
```

Os artefatos finais ficarão em `artifacts/best_model/`. Logs estão em `artifacts/logs/`.

## Como rodar a API localmente

```bash
source venv/bin/activate
uvicorn api.inference:app --reload --port 8000
```

## Ferramentas de dataset

### 1. Corrigir JSONL gerado por IA

```bash
python scripts/format_dataset.py --input dataset.jsonl --output dataset_tmp.jsonl
mv dataset_tmp.jsonl dataset.jsonl
```

### 2. Validar e gerar estatísticas

```bash
python scripts/prepare_dataset.py
```

### 3. Enviar para o GCS (opcional, requer credenciais)

```bash
python scripts/prepare_dataset.py --upload
```

### 4. Manual de preparação do dataset 

# 1) Ative a venv, se ainda não estiver ativa
source venv/bin/activate

# 2) Formate o arquivo gerado pela IA (saída em dataset_tmp.jsonl)
python scripts/format_dataset.py --input dataset.jsonl --output dataset_tmp.jsonl

# 3) Substitua o arquivo antigo pelo formatado
mv dataset_tmp.jsonl dataset.jsonl

# 4) Rode a validação/estatísticas
python scripts/prepare_dataset.py

# 5) (Opcional) Faça upload para o GCS quando quiser publicar
python scripts/prepare_dataset.py --upload

### 5. Resultado da execução do script de preparo do dataset será algo como:
🚀 Using device: mps
2025-10-28 12:28:32,160 - trainer.dataset_utils - INFO - 📁 Usando dataset local em data/conversations.jsonl
2025-10-28 12:28:32,161 - dataset-prep - INFO - 📄 Dataset validado em /Users/pedro.torres/Documents/projects/pingfy_ia/data/conversations.jsonl
2025-10-28 12:28:32,161 - dataset-prep - INFO - 📝 Mensagens de usuário: 194
2025-10-28 12:28:32,161 - dataset-prep - INFO - 🏷️  Intents únicas: 6
2025-10-28 12:28:32,161 - dataset-prep - INFO - 🏷️  Distribuição das intents: {'GREETING': 16, 'LEAD_INTENT': 51, 'PAYMENT': 48, 'SUPPORT': 61, 'CANCELATION': 23, 'FOLLOW_UP': 12}
## A idéia é que haja uma distruibuição/quantidade ampla de intents que vão ser utilizadas para alimentar as estatísticas finais ao usuário.
