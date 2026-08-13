# Coding Assistant

A local coding assistant built by fine-tuning **Qwen2.5-Coder-1.5B** on the **CodeAlpaca** dataset, then running it fully offline through **Ollama** and **Open WebUI**.

## How it works

1. **Train** — Fine-tune Qwen2.5-Coder-1.5B on CodeAlpaca using QLoRA (`train.ipynb`)
2. **Merge** — Merge the fine-tuned adapter into the base model (`merge_with_peft.py`)
3. **Convert** — Convert the merged model to GGUF format using llama.cpp
4. **Run** — Load it into Ollama using the `Modelfile`, and chat with it through Open WebUI

## Files

- `train.ipynb` — fine-tuning notebook
- `inference.ipynb` — testing the model
- `merge_with_peft.py` — merges the LoRA adapter into the base model
- `Modelfile` — Ollama config for running the model
- `final_model/` — the trained adapter

## Running it yourself

```bash
# 1. Merge the adapter into the base model
python merge_with_peft.py

# 2. Convert to GGUF using llama.cpp
python convert_hf_to_gguf.py ./merged_qwen_model --outfile qwen-finetuned.gguf --outtype q8_0

# 3. Create and run the model in Ollama
ollama create coding-assistant -f Modelfile
ollama run coding-assistant
```

Optionally, connect [Open WebUI](https://github.com/open-webui/open-webui) to your local Ollama instance for a chat UI.

## Built with
Python, PyTorch, Transformers, PEFT, llama.cpp, Ollama, Open WebUI