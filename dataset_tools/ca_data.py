#!/usr/bin/python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline)
from dolly.training.generate import generate_response
from os import listdir
import pandas
import torch

# Load the model to do our magic

candidate_models = [
    "databricks/dolly-v2-7b",
    "databricks/dolly-v2-3b",
]

instruct_pipeline = None

for model in candidate_models:
    try:
        instruct_pipeline = pipeline(model=model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    except Exception as e:
        print(f"Error {e} loading {model}")

if instruct_pipeline is None:
    raise Exception("Could not load any model")

# Load some strings we know the current model puts in appeals that are bad right away
with open("bad_appeal_strings.txt") as f: bad_appeal_strings = f.read().split("\n")

def load_data(path):
    imr = pandas.read_csv(
        path,
        usecols=["Determination", "TreatmentCategory", "TreatmentSubCategory",
                 "DiagnosisCategory", "DiagnosisSubCategory", "Type", "Findings",
                 "ReferenceID"
                 ],
        dtype=str)

    filtered_imr = imr[imr["Determination"].str.contains("Overturned")]
    return filtered_imr

imrs = load_data("./data_sources/ca-independent-medical-review-imr-determinations-trends-utf8.csv")

def generate_prompts(imr):
    determination = imr["Determination"]
    treatment = imr["TreatmentSubCategory"] or imr["TreatmentCategory"]
    findings = imr["Findings"]
    type = imr["Type"]
    print(determination)
    print(findings)
    generate_denial = f"What was the reason that {treatment} was originally denied in {findings}."
    generate_denial2 = f"Write a health insurance denial for {treatment} on the grounds of {type}."
    generate_appeal = f"The denial of {treatment} procedure was overturned in {findings}. Write an appeal for {treatment}."
    return [generate_denial, generate_denial2, generate_appeal]


prompts = generate_prompts(imrs.iloc[0])

for prompt in prompts:
    print(instruct_pipeline(prompt))
