from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer)
from dolly.training.generate import generate_response
from transformers import BitsAndBytesConfig
import torch
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model


instructions = [
    "Write a health insruance appeal."
]

pretrained_model_name_or_path = "./dolly/new_model"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="left")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,
    quantization_config=quantization_config,
)


model, tokenizer = load_model_tokenizer_for_generate(dbfs_output_dir)

# Use the model to generate responses for each of the instructions above.
for instruction in instructions:
    response = generate_response(instruction, model=model, tokenizer=tokenizer, **pipeline_kwargs)
    if response:
        print(f"Instruction: {instruction}\n\n{response}\n\n-----------\n")
