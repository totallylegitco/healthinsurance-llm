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
    r"""Summary:\s*(The|An|A)\s*\w+\s*[^.]*(requested|required|asked|requires|reimbursement)\s*[^.]*(for|medication)\s+([^.]+?)\.""",
    re.IGNORECASE)
alt_treatment_regex = re.compile(
    r"""At issue\s*(in this case|)\s*(is|)\s*(whether|if) ([^.]+?) is medically necessary""",
    re.IGNORECASE)

def get_treatment_from_imr(imr):
    findings = imr["Findings"]
    result = treatment_regex.search(findings)
    alt_result = alt_treatment_regex.search(findings)
    if result is not None:
        return result.group(4)
    elif alt_result is not None:
        return alt_result.group(4)
    else:
        print(f"No match in {findings}")
    return imr["TreatmentSubCategory"] or imr["TreatmentCategory"]


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


    def sketchy_sentence_filter(sentence):
        if "I am a" in sentence:
            return False
        if "agrees with the reviewer's findings" in sentence:
            return False
        if "The reviewer " in sentence:
            return False
        return True

    def generate_prompts(imr):
        print(imr)
        determination = imr["Determination"]
        treatment = get_treatment_from_imr(imr)
        diagnosis = imr["DiagnosisSubCategory"] or imr["DiagnosisCategory"]
        findings = imr["Findings"]
        grounds = imr["Type"]
        index = imr["ReferenceID"]
        rejection_prompts = [
            f"What was the reason that {treatment} was originally denied in {findings}.",
            f"Write a health insurance denial for {treatment} for diagnosis {diagnosis} on the grounds of {grounds}.",
            f"Deny coverage for {treatment} for {diagnosis}",
            f"Write a denial for {treatment}.",
            f"Expand on \"{treatment} is not medically necessary for {diagnosis}.\"",
        ]
        appeal_prompts = [
            f"The denial of {treatment} procedure was overturned in {findings}. Write an appeal for {treatment}.",
            f"The denial of {treatment} procedure was overturned in {findings}. Write an appeal for {treatment} for {diagnosis}.",
            f"Refute \"{treatment} is not medically necessary for {diagnosis}.\"",
        ]

        return (index, rejection_prompts, appeal_prompts)

    def cleanup_appeal(text):
        sentences = text.split(".")
        less_sketchy = ".".join(filter(sketchy_sentence_filter, sentences))
        if len(less_sketchy) < 40:
            return None
        if (not "Dear" in less_sketchy) and not ("To Whom" in less_sketchy):
            less_sketchy = f"Dear [INSURANCECOMPANY];\n{less_sketchy}"
        return less_sketchy

    def cleanup_rejections(text):
        if not "[MEMBER]" in text:
            text = f"Dear [MEMBER]; {text}."
        if not "appeal" in text:
            text = f"{text}. You have the right to appeal this decision."
        return text



    l = imrs.iloc[0:20].apply(generate_prompts, axis=1).tolist()
    for (idx, rejection_prompts, appeal_prompts) in l:
        results = instruct_pipeline(rejection_prompts + appeal_prompts)
        rejections = map(cleanup_rejection, results[0:len(rejection_prompts)])
        appeals = map(cleanup_appeal, results[len(rejection_prompts):])
        i = 0
        for r in rejections:
            i = i + 1
            with open(join(gen_loc, f"{idx}MAGIC{i}_rejection.txt")) as f:
                f.write(r)
        for a in appeals:
            if a is None:
                continue
            i = i + 1
            with open(join(gen_loc, f"{idx}MAGIC{i}_appeal.txt")) as f:
                f.write(a)




def work_with_biogpt():
    instruct_pipeline = pipeline(model="microsoft/BioGPT-Large-PubMedQA", max_new_tokens=100)


    def generate_biogpt_hacks(imr):
        treatment = get_treatment_from_imr(imr)
        diagnosis = imr["DiagnosisSubCategory"] or imr["DiagnosisCategory"]
        findings = imr["Findings"]
        index = imr["ReferenceID"]
        return (index, f"{treatment} is medically necessary for {diagnosis} because")
    l = imrs.iloc[0:20].apply(generate_biogpt_hacks, axis=1).tolist()
    idxs = map(lambda r: r[0], l)
    qs = map(lambda r: r[1], l)
    transformed = instruct_pipeline(list(qs))
    joined = zip(idxs, transformed)

    def write_result(res):
        idx = res[0]
        reason = res[1]
        with open(join(gen_loc, f"{idx}MAGICB_appeal.txt")) as o:
            o.write("""Dear [InsuranceCompany];

I am writing you to appeal claim [CLAIMNUMBER]. I believe that it is medically necessary.""")
            o.write(reason)
            o.write("\n")
            o.write("Sincerely,\n[YOURNAME]")

    for res in joined:
        write_result(res)


#work_with_dolly()
work_with_biogpt()
