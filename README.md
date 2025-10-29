# TicketSense-AI

TicketSense-AI é o módulo de inteligência que classifica automaticamente as intenções que chegam pelos canais de atendimento digital da empresa. Ele lê conversas, identifica o motivo do contato (ex.: pagamento, cancelamento, suporte) e entrega essa resposta para o ecossistema PingFy (futuramente) com rapidez e rastreabilidade.

## Por que este módulo existe
- Padronizar a triagem de tickets e conversas, reduzindo tempo de atendimento humano.
- Garantir histórico fiel de métricas de intenção para relatórios e alertas.
- Permitir que a equipe utilize um modelo treinável e auditável, sem depender de serviços externos.

## Como o projeto está organizado
```
pingfy_ia/
├── api/             → API FastAPI que expõe as previsões do modelo.
├── config/          → Configurações compartilhadas (device, GCS, API, treino).
├── trainer/         → Pipeline completo de treinamento e utilidades.
├── scripts/         → Ferramentas rápidas para cuidar do dataset.
├── artifacts/       → Modelos treinados, checkpoints e logs gerados.
├── data/            → Cópia local do dataset validado (não versionado).
├── models/          → Espaço para exportações extras do modelo.
├── requirements.txt → Dependências Python.
└── README.md        → Este guia.
```

### O que cada parte faz
- `api/inference.py`: carrega o melhor modelo disponível e disponibiliza os endpoints `/predict_intent`, `/predict_batch`, `/health` e `/model_info`.
- `api/schemas.py`: define os formatos de entrada e saída da API, para facilitar integrações.
- `config/config.py`: concentra todas as configurações (como diretórios, credenciais e device) em dataclasses simples.
- `trainer/train.py`: guia principal do treinamento; prepara dados, ajusta o modelo base (DistilBERT), calcula métricas e salva artefatos.
- `trainer/dataset_utils.py`: valida o dataset, cuida dos splits, faz encode das intenções e fornece o `Dataset` usado no treino.
- `trainer/model_utils.py`: cria o modelo/tokenizer, mantém o arquivo `label_map.json` e envia resultados para o Google Cloud Storage, se credenciais estiverem configuradas.
- `scripts/format_dataset.py`: corrige arquivos JSONL gerados por IA, garantindo um item por linha.
- `scripts/prepare_dataset.py`: valida o dataset, mostra estatísticas de distribuição e, opcionalmente, envia o arquivo para o GCS.

## Fluxo de trabalho recomendado
1. **Organize o dataset** com conversas rotuladas em JSONL (um registro por linha).
2. **Formate e valide** o arquivo usando os scripts de apoio.
3. **Treine o modelo** com `python -m trainer.train` para gerar novos artefatos em `artifacts/`.
4. **Suba a API** com o modelo mais recente e integre com o restante da plataforma.

## Preparação do ambiente
- Pré-requisitos: Python 3.10+, `pip`, acesso opcional ao Google Cloud Storage (caso use upload automático).
- Sugestão para criar o ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Comandos rápidos
- **Aplicar formatação no dataset bruto**
  ```bash
  python scripts/format_dataset.py --input dataset.jsonl --output dataset_tmp.jsonl
  mv dataset_tmp.jsonl dataset.jsonl
  ```

- **Validar dataset, gerar estatísticas e copiar para `data/`**
  ```bash
  python scripts/prepare_dataset.py
  ```

- **Enviar dataset validado para o Google Cloud Storage** (requer credencial configurada)
  ```bash
  python scripts/prepare_dataset.py --upload
  ```

- **Rodar o treinamento completo**
  ```bash
  python -m trainer.train
  ```

- **Iniciar a API de inferência no modo desenvolvimento**
  ```bash
  uvicorn api.inference:app --reload --port 8000
  ```

- **Consultar manualmente a API (exemplo)**
  ```bash
  curl -X POST http://localhost:8000/predict_intent \
       -H "Content-Type: application/json" \
       -d '{"text": "Olá, preciso renegociar minha fatura"}'
  ```
- **Para conseguir ver todas as probabilidades de intenção, adicione o parâmetro "return_probabilities": true no final**
  ```bash
  curl -X POST http://localhost:8000/predict_intent \
     -H "Content-Type: application/json" \
     -d '{"text": "Tenho cobrança duplicada e quero saber como faço para resolver isso.", "return_probabilities": true}'
  ```
## API de inferência
- `POST /predict_intent`: recebe um texto e devolve a intenção mais provável com a confiança.
- `POST /predict_batch`: aceita uma lista de textos para classificação em lote.
- `GET /health`: informa se o modelo está carregado e em qual device está rodando.
- `GET /model_info`: descreve o modelo ativo, classes conhecidas e tamanho máximo das mensagens.

## Scripts auxiliares
- `format_dataset.py`: usado logo após receber um JSON bruto vindo de uma geração automática. Ele garante que cada item estará em uma linha e pronto para ser lido pelo restante do pipeline.
- `prepare_dataset.py`: confirma que o arquivo está válido, exibe total de mensagens por intenção e copia para `data/`. Com a flag `--upload`, realiza o envio para o bucket definido em `config/config.py`.

## Configuração por ambiente
Use um arquivo `.env` na raiz (não versionado) para guardar:

```
GCS_BUCKET_NAME=nome-do-bucket
GCS_DATASET_PATH=data/conversations.jsonl
GCS_MODEL_PATH=models/checkpoints/distilbert-intents
GOOGLE_APPLICATION_CREDENTIALS=/caminho/para/credencial.json
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false
```

Se nenhuma variável for definida, o projeto usa os valores padrão presentes em `config/config.py`.

## Formato esperado do dataset
- Arquivo `.jsonl`, com uma conversa por linha.
- Cada conversa contém uma lista `messages`, e o script consome apenas itens onde `role` é `"user"`.
- A intenção deve estar na chave `intent`. Pode ser uma lista com mais de um rótulo (multi-intenção).

Exemplo:

```jsonl
{"conversation_id": "conv_001", "channel": "whatsapp", "messages": [
  {"role": "user", "text": "Oi, preciso da segunda via do boleto", "intent": ["PAYMENT"]},
  {"role": "agent", "text": "Claro, já envio o link."}
]}
```

## Artefatos gerados
- `artifacts/best_model/`: contém o modelo final, tokenizer e arquivo `label_map.json` com o dicionário das intenções.
- `artifacts/checkpoints/`: guarda os checkpoints intermediários do treinamento.
- `artifacts/logs/training.log`: registra cada execução de treino para auditoria.
- `models/`: espaço reservado para exportações adicionais (ex.: TorchScript, ONNX) caso o time precise.

Licença: consulte `LICENSE.txt` (Business Source License 1.1). Para uso comercial em produção, é necessário acordo com PaletotCode.
