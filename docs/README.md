# TicketSense-AI — Documentação Oficial

> Versão 1.0 · Atualizado em 01/11/2025  
> Contato técnico: equipe@ticketsense.ai

---

## Índice
1. [Visão Geral](#visão-geral)
2. [Arquitetura do Projeto](#arquitetura-do-projeto)
3. [Preparação do Ambiente](#preparação-do-ambiente)
4. [Fluxo Operacional](#fluxo-operacional)
5. [Detalhes por Componente](#detalhes-por-componente)
6. [Painel Administrativo (opcional)](#painel-administrativo-opcional)
7. [Qualidade e Avaliação](#qualidade-e-avaliação)
8. [Diretrizes de Contribuição](#diretrizes-de-contribuição)
9. [Glossário de Termos](#glossário-de-termos)

---

## Visão Geral

TicketSense-AI é uma plataforma de **classificação multi-intenção** com foco em atendimento comercial. Ela identifica, em tempo real, o motivo de contato de um cliente e fornece até três intenções ordenadas (com probabilidades) — base para copilotos de vendas, análise de leads e automação de respostas.

O MVP atual entrega:
- **Classificação de intenções** com modelo `microsoft/mdeberta-v3-base` fine-tunado.
- **Geração sintética** (Gemini 2.0) com receita focada em leads.
- **Painel opcional** para orquestrar geração, treino, histórico e comandos.
- **Ferramentas de qualidade** (auditoria, avaliação) e pipelines padronizados via `Makefile`.

---

## Arquitetura do Projeto

```
pingfy_ia/
├── api/                     # FastAPI (inferência + endpoints administrativos)
├── config/                  # Configurações centralizadas (.env → dataclasses)
├── docs/                    # Documentação (este diretório)
├── scripts/                 # CLIs de geração, validação, auditoria, avaliação
│   └── utils/               # Receita, clients LLM, prompt helpers
├── trainer/                 # Pipeline de treinamento (dataset, modelo, loop HF)
├── artifacts/               # Modelos, logs, checkpoints (gerados em runtime)
├── data/                    # Datasets locais (.jsonl) (gerados em runtime)
├── requirements.txt         # Dependências Python
├── Makefile                 # Comandos padronizados
└── MANUAL_USO.md            # Guia rápido (para mão na massa)
```

Componentes-chave:
- **API FastAPI** (`api/`): responde `/predict_intent` com top‑3 intents e expose endpoints de administração.
- **Scripts** (`scripts/`) para geração (Gemini), validação, auditoria e avaliação.
- **Trainer** (`trainer/`) com HuggingFace Trainer + utilidades (dataset, tokenizer, modelo).
- **Receita** (`scripts/utils/dataset_recipe.py`) com ~15k amostras direcionadas a LEAD_INTENT.
- **Painel** (opcional, em `admin_control_panel/`, fora deste documento).

---

## Preparação do Ambiente

### Requisitos mínimos
- macOS (Apple Silicon M-series) ou Linux x86_64.
- Python 3.10+; Node.js apenas para o painel opcional.
- Conta Google com acesso ao Gemini 2.0 free tier.

### Passos iniciais
```bash
git clone <repo>
cd pingfy_ia
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Variáveis de ambiente (.env)
Coloque na raiz (`.env`):
```env
GEMINI_API_KEY=seu_token
OPENAI_API_KEY=opcional
GCS_BUCKET_NAME=pingfy-dataset
GCS_DATASET_PATH=data/synthetic_dataset.jsonl
```
Se precisar de GCS upload: `GOOGLE_APPLICATION_CREDENTIALS=/caminho/para/service-account.json`.

---

## Fluxo Operacional

O `Makefile` padroniza os passos. A tabela abaixo resume:

| Comando | Ação |
|---------|------|
| `make env` | Cria venv + instala deps |
| `make dataset` | Gera dataset do zero |
| `make resume` | Retoma geração (usa arquivo atual) |
| `make validate` | `prepare_dataset.py --analysis` |
| `make quality` | Auditoria (`analise_qualidade_dataset.py`) |
| `make train` | Treino completo (`LOCAL_DATASET_PATH=...`) |
| `make eval` | Avaliação (`evaluate_model.py --threshold 0.4`) |
| `make api` | Sobe FastAPI com reload |
| `make clean` | Remove venv/checkpoints (cautela) |

Fluxo típico (end-to-end):
1. `make env` — primeiro uso.
2. `make resume` — gera dataset `data/synthetic_dataset_v2.jsonl`.
3. `make validate` e/ou `make quality` — garante distribuição e qualidade.
4. `make train` — fine-tuning do modelo mDeBERTa (salva em `artifacts/best_model/`).
5. `make eval` — checa métricas (F1, hit@3).
6. `make api` — serve modelo em `http://localhost:8000`.

---

## Detalhes por Componente

### 1. Configuração (`config/`)
- **config.py**: carrega `.env`, define `ModelConfig`, `TrainingConfig`, `APIConfig`, `GCSConfig`.  
  - `model_name = "microsoft/mdeberta-v3-base"`  
  - Intents padrão incluem `LEAD_INTENT` como primeira classe.
- `DEVICE` seleciona automaticamente MPS / CUDA / CPU.

### 2. Dataset sintético (`scripts/utils/`)
- **dataset_recipe.py**: lista de `GenerationTask` (singles e combinações) totalizando 15k amostras.  
  - LEAD_INTENT e combos com UPGRADE/INFORMATION/PAYMENT têm prioridade.  
  - Técnicos (TECHNICAL_ISSUE) reforçados com COMPLAINT/SUPPORT.  
- **generate_synthetic_dataset.py**:  
  - CLI com `--resume`, `--analysis`, logging e validação.  
  - `get_client` usa `GeminiClient` (`gemini-2.0-flash-lite`) por padrão.  
  - Salva incrementalmente e retoma facilmente.
- **generation_utils.py**: prompt padrão, parser incremental (`iter_json_objects`), validação de sample.
- **llm_clients.py**:  
  - `GeminiClient`, `OpenAIClient`, `MockClient`.  
  - Controle de temperatura, fallback se a primeira chamada falhar.

### 3. Scripts operacionais (`scripts/`)
- **prepare_dataset.py**: valida/analisa dataset local (`--dataset`, `--analysis`).  
- **analise_qualidade_dataset.py**: relatório de variabilidade, duplicatas, baixa qualidade, exporta JSON/CSV.  
- **evaluate_model.py**: rodar pós-treino (`--threshold`, `--topk`); inclui LEAD combos.

### 4. Treinador (`trainer/`)
- **dataset_utils.py**:  
  - `ensure_dataset()` (; evita download GCS sem credenciais).  
  - `load_dataset()` parse JSONL (com buffer para linhas quebradas).  
  - `select_indices()` com fallback quando estratificação falha.  
  - `IntentDataset` wrapper (torch tensor).
- **model_utils.py**:  
  - `create_tokenizer(use_fast=False)` para mDeBERTa (evita conversão tiktoken).  
  - `create_model()` carrega `AutoModelForSequenceClassification` com `problem_type="multi_label_classification"`.
- **train.py**:  
  - Logging em `artifacts/logs/training.log`.  
  - `TrainingArguments`: batch size 1 + grad accumulation (controlado por config).  
  - Sem `gradient_checkpointing` (maior estabilidade em MPS).  
  - Salva melhor modelo em `artifacts/best_model/`, label map e tokenizer.

### 5. API (`api/`)
- **inference.py**:  
  - Carrega best_model + tokenizer no startup.  
  - `/predict_intent`: retorna `intent`, `confidence`, lista `intents` (top‑3 com threshold 0.35), `all_probabilities` opcional.  
  - `/predict_batch`, `/health`, `/model_info`.  
- **schemas.py**: modelos Pydantic (requests/responses).  
- **admin.py**:  
  - `/admin/dashboard`, `/admin/training/history`, `/admin/models`, `/admin/models/activate`.  
  - `/admin/training/start` + `/admin/training/stream` (SSE).  
  - `/admin/commands/run`: executa comandos locais (auto usa `sys.executable` para python).  
  - `TrainingManager` gerencia fila, broadcast de logs, parse de métricas.

### 6. Painel (opcional)
Repositório possui `admin_control_panel/` (React + Vite + Tailwind). Funcionalidades:
  - Dashboard com gráficos de evolução (`MetricTrends`), datasets e playground de inferência.  
  - Abas: Treinamento (logs em tempo real), Modelos (ativar best_model), Histórico (tabela), Automação (biblioteca de comandos), Configurações (variáveis rápidas).  
  - Depende dos endpoints `/admin`. Não é obrigatório para usar a API core.

---

## Qualidade e Avaliação

### Auditoria (qualidade sintética)
```bash
make quality
# outputs:
# - artifacts/logs/dataset_quality.json
# - artifacts/logs/dataset_quality.csv
```

### Avaliação do modelo
```bash
make eval  # threshold 0.4, topk 3
```
Métricas esperadas (último treino):
- Subset accuracy ≈ 0.56–0.58 (teste fixo)
- Hit@3 ≈ 0.95
- F1 micro ≈ 0.68 (threshold 0.5) / 0.7+ com threshold calibrado
- LEAD_INTENT recall 1.0; technical/commercial combos acima de 0.7 F1.

---

## Diretrizes de Contribuição

1. Use `make env` + `make resume` antes de mexer.  
2. Atualize receita (`dataset_recipe.py`) com cuidado — mantenha `TOTAL_SAMPLES`.  
3. Sempre rode `make validate` + `make train` + `make eval` antes de subir pull request.  
4. Documente métricas novas no `MANUAL_USO.md` e, se for relevante, adaptações no painel.
5. Para novos scripts, anexe instruções no manual e considere ganhar um atalho no Makefile.

---

## Glossário de Termos

| Termo | Significado |
|-------|-------------|
| **Intenção (Intent)** | Motivo de contato identificado pela IA (ex.: `LEAD_INTENT`, `PAYMENT`). |
| **Multi-intenção** | Quando uma mensagem tem mais de uma intenção relevante (ex.: `["PAYMENT", "SUPPORT"]`). |
| **LEAD_INTENT** | Intenção que identifica um potencial comprador ou oportunidade de venda. |
| **Dataset sintético** | Conjunto de dados gerado artificialmente (neste caso, via Gemini) seguindo nossa receita. |
| **Gemini 2.0 flash-lite** | Modelo LLM gratuito do Google usado para gerar exemplos sintéticos. |
| **Recipe** | Arquivo que define quantas amostras gerar por intenção/combo (`dataset_recipe.py`). |
| **Resume** | Recuperar geração interrompida; o script lê o arquivo existente e calcula o saldo. |
| **MDEBERTa-V3-Base** | Modelo transformer da Microsoft, usado como backbone para classificação. |
| **Gradient accumulation** | Técnica para simular batch maior acumulando gradientes antes do update. |
| **Subset accuracy** | Métrica que só conta acerto quando todas as intenções da mensagem foram previstas corretamente. |
| **Hit@K** | Percentual de casos em que a intenção correta aparece entre as K sugestões principais. |
| **F1 micro** | Média harmônica de precisão e recall considerando todos os rótulos (multi-label). |
| **SSE (Server-Sent Events)** | Protocolo para enviar logs/status contínuos do backend para o painel. |
| **Pipeline** | Sequência de etapas (geração → validação → treino → avaliação → inferência). |
| **Makefile** | Arquivo que define comandos rápidos (`make <alvo>`) para automatizar tarefas. |
| **Painel admin** | Interface web opcional que consome os endpoints `/admin/` para monitorar/acionar tarefas. |

---

**Dúvidas?** Abra uma issue ou contate a equipe técnica. Esta documentação deve ser mantida em sincronia com o código-fonte — sinta-se à vontade para propor melhorias. 🚀
