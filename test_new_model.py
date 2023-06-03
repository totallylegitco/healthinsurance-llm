from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline)
from dolly.training.generate import generate_response
import torch
from os import listdir
from os.path import isfile, join


pretrained_model_name_or_path = "./dolly/new_model"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="left")

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

old_instruct_pipeline = pipeline(model="databricks/dolly-v2-7b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")


raw_dataset = "appeals-llm-data"

data_files = [f for f in listdir(raw_dataset) if (isfile(join(raw_dataset, f)))]

rejection_files = [f for f in data_files if f.endswith("_rejection.txt")]

for f in rejection_files:
        print(f)
        with open(join(raw_dataset, f), encoding="utf-8") as r: print(generate_response("I received a denial: " + r.read() + ". Write an appeal to this denial", model=model, tokenizer=tokenizer))
        print(f)
        with open(join(raw_dataset, f), encoding="utf-8") as r: print(old_instruct_pipeline("I received a denial: " + r.read() + ". Write an appeal to this denial"))
