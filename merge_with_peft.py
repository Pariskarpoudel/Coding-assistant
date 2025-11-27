from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json

print("="*60)
print("Merging LoRA Adapter with Base Model (Dequantized)")
print("="*60)

# Check GPU
if torch.cuda.is_available():
    print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠ No GPU detected - this will be SLOW on CPU")

print("\nStep 1: Loading adapter configuration...")
adapter_path = "./final_model2"
config = PeftConfig.from_pretrained(adapter_path)
print(f"✓ Base model: {config.base_model_name_or_path}")

# CRITICAL FIX: Use unquantized BASE model (not Instruct)
# You trained on: unsloth/Qwen2.5-Coder-1.5B-bnb-4bit (base model)
# So use the unquantized base version:
UNQUANTIZED_BASE = "Qwen/Qwen2.5-Coder-1.5B"
print(f"✓ Using unquantized base: {UNQUANTIZED_BASE}")

print("\nStep 2: Loading base model in fp16 (NO quantization)...")
base_model = AutoModelForCausalLM.from_pretrained(
    UNQUANTIZED_BASE,  # Changed from config.base_model_name_or_path
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    # CRITICAL: Load without any quantization
    load_in_8bit=False,
    load_in_4bit=False
)
print("✓ Base model loaded in fp16")

print("\nStep 3: Loading adapter...")
model = PeftModel.from_pretrained(
    base_model, 
    adapter_path,
    torch_dtype=torch.float16
)
print("✓ Adapter loaded")

print("\nStep 4: Merging adapter with base model...")
merged_model = model.merge_and_unload()
print("✓ Merge complete!")

print("\nStep 5: Saving merged model...")
output_path = "./merged_qwen_model"
os.makedirs(output_path, exist_ok=True)

# Save model
merged_model.save_pretrained(
    output_path,
    safe_serialization=True,  # Use safetensors format
    max_shard_size="5GB"
)

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(UNQUANTIZED_BASE)  # Changed from adapter_path
tokenizer.save_pretrained(output_path)

# CRITICAL FIX: Clean up config.json to remove quantization metadata
config_path = os.path.join(output_path, "config.json")
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    # Remove quantization-related keys
    keys_to_remove = [
        'quantization_config',
        'load_in_8bit',
        'load_in_4bit',
        'bnb_4bit_compute_dtype',
        'bnb_4bit_use_double_quant',
        'bnb_4bit_quant_type'
    ]
    
    removed = []
    for key in keys_to_remove:
        if key in model_config:
            del model_config[key]
            removed.append(key)
    
    if removed:
        print(f"\n✓ Removed quantization config: {', '.join(removed)}")
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)

print(f"\n✓ Saved to: {output_path}")

print("\n" + "="*60)
print("SUCCESS! Model merged and cleaned!")
print("="*60)
print("\nNext steps:")
print("1. Convert to GGUF:")
print(f"   python convert_hf_to_gguf.py {output_path} --outfile D:/qwen-finetuned.gguf --outtype f16")
print("\n2. If conversion still fails, try:")
print(f"   python convert_hf_to_gguf.py {output_path} --outfile D:/qwen-finetuned.gguf --outtype q8_0")
print("="*60)