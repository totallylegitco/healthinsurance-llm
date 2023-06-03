#!/usr/bin/python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline)
from dolly.training.generate import generate_response
from os import listdir
import pandas
import torch

import re

gen_loc = "generated-llm-data"

treatment_regex = re.compile(
    r"""Summary:\s*(The|An)\s*\w+\s*[^.]*(requested|required|asked|requires)\s*[^.]*for\s+([^.]+?)\.""",
    re.IGNORECASE)

def get_treatment_from_imr(imr):
    treatment = None
    findings = imr["Findings"]
    result = treatment_regex.search(findings)
    if result is not None:
        treatment = result.group(3)
    else:
        print(f"No match in {findings}")
    return treatment  or imr["TreatmentSubCategory"] or imr["TreatmentCategory"]


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

def work_with_dolly():
    # Load the model to do our magic

    candidate_models = [
        "databricks/dolly-v2-7b",
        "databricks/dolly-v2-3b",
    ]

    instruct_pipeline = None

    for model in candidate_models:
        try:
            instruct_pipeline = pipeline(model=model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
            break
        except Exception as e:
            print(f"Error {e} loading {model}")

    if instruct_pipeline is None:
        raise Exception("Could not load any model")


    def generate_prompts(imr):
        print(imr)
        determination = imr["Determination"]
        treatment = get_treatment_from_imr(imr)
        diagnosis = imr["DiagnosisSubCategory"] or imr["DiagnosisCategory"]
        findings = imr["Findings"]
        grounds = imr["Type"]
        index = imr["ReferenceID"]
        return [
            f"What was the reason that {treatment} was originally denied in {findings}.",
            f"Write a health insurance denial for {treatment} for diagnosis {diagnosis} on the grounds of {grounds}.",
            f"Write a for {treatment} for diagnosis {diagnosis} on the grounds of {type}.",
            f"The denial of {treatment} procedure was overturned in {findings}. Write an appeal for {treatment}.",
            f"The denial of {treatment} procedure was overturned in {findings}. Write an appeal for {treatment} for {diagnosis}.",
            f"Deny coverage for {treatment} for {diagnosis}",
            f"Deny coverage for {treatment}",
            f"Write a denial for {treatment}.",
            f"Expand on \"{treatment} is not medically necessary for {diagnosis}.\"",
            f"Refute \"{treatment} is not medically necessary for {diagnosis}.\""
            f"Summarize {findings}",
        ]


    prompts = generate_prompts(imrs.iloc[0])
    results = instruct_pipeline(prompts)

    joined = zip(prompts, results)

    for (prompt, result) in joined:
        print(prompt)
        print("\n")
        print(result)
        print("\n")

def work_with_biogpt():
#    tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large-PubMedQA")

#    model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large-PubMedQA")
    instruct_pipeline = pipeline(model="microsoft/BioGPT-Large-PubMedQA", max_new_tokens=100)


    def generate_biogpt_hacks(imr):
        print(imr)
        treatment = get_treatment_from_imr(imr)
        diagnosis = imr["DiagnosisSubCategory"] or imr["DiagnosisCategory"]
        findings = imr["Findings"]
        index = imr["ReferenceID"]
        return (index, f"{treatment} is medically necessary for {diagnosis} because")
    l = imrs.iloc[0:20].apply(generate_biogpt_hacks, axis=1).tolist()
    idxs = map(lambda r: r[0], l)
    qs = map(lambda r: r[1], l)
    transformed = instruct_pipeline(qs)
    joined = zip(idxs, transformed)

    def write_result(res):
        idx = res[0]
        reason = res[1]
        with open(join(gen_loc, f"{idx}_appeal2.txt")) as o:
            o.write("""Dear [InsuranceCompany];

I am writing you to appeal claim [CLAIMNUMBER]. I believe that it is medically necessary.""")
            o.write(reason)
            o.write("\n")
            o.write("Sincerely,\n[YOURNAME]")

    for res in joined:
        write_result(res)


#work_with_dolly()
work_with_biogpt()
