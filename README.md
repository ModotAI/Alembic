# ⚗ Alembic

**TUI tool for synthetic dataset generation and LLM distillation.**

Alembic distills knowledge from large language models into structured datasets for training smaller models. Configure your API, pick topics, choose output format, and let it run.

![Rust](https://img.shields.io/badge/Rust-000000?style=flat&logo=rust&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

## Features

- **Multi-provider** — Groq, OpenRouter, Cerebras, Mistral, OpenAI, Claude
- **Synthetic Q&A generation** — AI generates questions on your topics, then answers them
- **Multiple output formats** — ChatML, Llama, Alpaca, or custom templates
- **Interactive TUI** — Configure everything visually with keyboard navigation
- **Rate limit handling** — Automatic retry with backoff on 429 errors
- **Resume-friendly** — Outputs are saved incrementally as JSONL

## Install

```bash
git clone https://github.com/ThingAI/alembic.git
cd alembic
cargo build --release
```

## Usage

```bash
./target/release/alembic
```

### TUI Controls

| Key | Action |
|-----|--------|
| `Tab` / `1-4` | Switch between sections |
| `↑` `↓` | Navigate fields |
| `Enter` | Edit field or cycle option |
| `Esc` | Cancel edit / go back |
| `q` | Quit |

### Sections

1. **Provider** — Select API provider, model, enter API key
2. **Topics** — Define topics, number of samples, language, system prompt
3. **Format** — Choose output format (ChatML/Llama/Alpaca), preview, output path
4. **Run** — Review summary and start distillation

## Example

Generate 1000 Italian Q&A pairs about culture and history:

**Provider:** Groq  
**Model:** `llama-3.1-8b-instant`  
**System Prompt:**
```
Rispondi in italiano in modo conciso e diretto, massimo 2-3 frasi. Solo fatti, niente premesse.
```
**Topics:**
```
storia italiana, arte italiana, cucina regionale, scienza italiana, letteratura italiana
```
**Samples:** 1000  
**Language:** it  
**Format:** ChatML  

Output (`dataset.jsonl`):
```json
{"text":"<|im_start|>user\nQual è il fiume più lungo d'Italia?<|im_end|>\n<|im_start|>assistant\nIl Po, con 652 km, è il fiume più lungo d'Italia. Nasce dal Monviso e sfocia nel Mare Adriatico.<|im_end|>","topic":"geografia italiana"}
```

## Supported Providers

| Provider | Free Tier | Signup |
|----------|-----------|--------|
| [Groq](https://console.groq.com) | 30 req/min, no daily limit | No card |
| [OpenRouter](https://openrouter.ai) | 20 req/min, 50 req/day | No card |
| [Cerebras](https://cloud.cerebras.ai) | 30k TPM | Card required |
| [Mistral](https://console.mistral.ai) | Limited | No card |
| [OpenAI](https://platform.openai.com) | Pay-as-you-go | Card required |
| [Anthropic](https://console.anthropic.com) | Pay-as-you-go | Card required |

## Output Formats

**ChatML:**
```
<|im_start|>user
What is the capital of Italy?<|im_end|>
<|im_start|>assistant
The capital of Italy is Rome.<|im_end|>
```

**Llama:**
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
What is the capital of Italy?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
The capital of Italy is Rome.<|eot_id|>
```

**Alpaca:**
```json
{"instruction": "What is the capital of Italy?", "input": "", "output": "The capital of Italy is Rome."}
```

## Architecture

```
src/
├── main.rs      — Entry point, TUI event loop + async task spawning
├── api.rs       — Multi-provider API client (Claude, OpenAI-compat)
├── distill.rs   — Question generation + answer distillation + save
└── tui.rs       — Ratatui interface (Setup → Running → Done)
```

## Why Alembic?

An alembic is the classic distillation apparatus — and that's exactly what this tool does: distill knowledge from large models into concentrated datasets for training smaller ones.

## License

Apache 2.0

## Author

Built by [ThingAI](https://github.com/ThingAI)
