#!/usr/bin/python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline)
from os.path import join
import pandas
import torch
import argparse
import re
import itertools
from .utils import *
import multiprocessing

flatten = itertools.chain.from_iterable

gen_loc = "generated-llm-data"

parser = argparse.ArgumentParser(
    prog='CA Data Generator',
    description='Generate data',
    epilog='Magic')

parser.add_argument(
    '--small-gpu',
    action='store_true')

args = parser.parse_args()

treatment_regex = re.compile(
    r"""\s*(The|An|A)?\s*(parent|father|mother|patient|enrollee|member|provider)\s*[^.]*(requested|required|asked|requires|reimbursement|coverage|requesting|has)\s*[^.]*(of|for|medication|reimbursement|coverage|services)\s+(?P<treatment>\d*\w+.+?)\.""",
    re.IGNORECASE)
alt_treatment_regex = re.compile(
    r"""At issue\s*(in this case|)\s*(is|)\s*(whether|if)\s+(?P<treatment>\d*\w+.+?) (is|were|was) medically (necessary|indicated)""",
    re.IGNORECASE)
more_alt_treatment_regex = re.compile(
    r"""the requested (medication|treatment|service|procedure)\s+(?P<treatment>\d*\w+.+?) (is|were|was) (likely to be|medically necessary|medically indicated)""",
    re.IGNORECASE)

even_more_alt_treatment_regex = re.compile(
    r"""(Therefore|Thus|As such),[^.]*?\s+(an|a|the|that) (?P<treatment>\w+[^.]+?) (is|were|was|should be) (medically necessary|medically indicated|likely to be|authorized)""",
    re.IGNORECASE)

perscribed_regex = re.compile(
    r"""patients provider has prescribed the medication\s+(?P<treatment>[^.]+?).""", re.IGNORECASE)

wishes_to_regex = re.compile(r"""(wishes|desires|like) to (undergo|take)\s+(?P<treatment>[^.]+?).""", re.IGNORECASE)

health_plan_not_necessary_regex = re.compile(r"""The (Health Plan|Plan|Insurance Company) (determined the|determined|indicates) (?P<treatment>.+?) (is|was|were) not""", re.IGNORECASE)

almost_sketchy_regex = re.compile(r"""treatment[^.]*with\s+(?P<treatment>[^.]+?) (is|were|was)""", re.IGNORECASE)

sketchy_regex = re.compile(r"""(requested|required|asked|requires|reimbursement|coverage|request|requesting)\s*[^.]*(for|medication|reimbursement|coverage|of)\s+(?P<treatment>\d*\w+.+?)\.""",
    re.IGNORECASE)

seeking_regex = re.compile(r"""is seeking (?P<treatement>) for (?P<diagnosis>[^.]+)""", re.IGNORECASE)
admitted_regex = re.compile(r"""admitted to the hospital for (?P<diagnosis>[^.]+)""", re.IGNORECASE)
recommended_regex = re.compile(r"""physicians recommended (?P<treatment>[^.]+)""", re.IGNORECASE)

treatment_regexes = [
    treatment_regex,
    alt_treatment_regex,
    more_alt_treatment_regex,
    perscribed_regex,
    wishes_to_regex,
    health_plan_not_necessary_regex,
    almost_sketchy_regex,
    sketchy_regex,
    seeking_regex,
    recommended_regex]

diagnosis_regexes = [
    seeking_regex,
    admitted_regex,
    ]

def get_treatment_from_imr(imr):
    findings = imr["Findings"]
    for r in treatment_regexes:
        matches = r.search(findings)
        if matches is not None:
            return matches.group("treatment")
    return imr["TreatmentSubCategory"] or imr["TreatmentCategory"]


def get_diagnosis_from_imr(imr):
    findings = imr["Findings"]
    for r in diagnosis_regexes:
        matches = r.search(findings)
        if matches is not None:
            return matches.group("diagnosis")
    return imr["DiagnosisSubCategory"] or imr["DiagnosisCategory"]

def extract_text(result):
    if result is None:
        return None
    if "generated_text" not in result[0]:
        return None
    return result[0]["generated_text"]

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

def work_with_generative():
    # Load the model to do our magic

    candidate_models = [
#        "ausboss/llama-30b-supercot",
#        "CalderaAI/30B-Lazarus",
#        "tiiuae/falcon-40b-instruct",
        "databricks/dolly-v2-12b",
        "databricks/dolly-v2-7b",
        "databricks/dolly-v2-3b",
    ]

    instruct_pipeline = None

    for model in candidate_models:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model)
            print(f"Loading {model}\n")
            if args.small_gpu:
                instruct_pipeline = pipeline(
                    model=model,
                    tokenizer=tokenizer,
                    eos_token_id=tokenizer.eos_token_id,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    model_kwargs={'load_in_8bit': True},
                    device_map="auto",
                    max_new_tokens=512
                )
            else:
                instruct_pipeline = pipeline(
                    model=model,
                    tokenizer=tokenizer,
                    eos_token_id=tokenizer.eos_token_id,
                    top_k = 10,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto",
                    max_new_tokens=512)
            break
        except Exception as e:
            print(f"Error {e} loading {model}")

    if instruct_pipeline is None:
        raise Exception("Could not load any model")


    def generate_prompts(imr):
        determination = imr["Determination"]
        treatment = get_treatment_from_imr(imr)
        diagnosis = get_diagnosis_from_imr(imr)
        findings = imr["Findings"].strip("\n")
        grounds = imr["Type"]
        index = imr["ReferenceID"]


        def append_context_full(prompt):
            if prompt is None:
                return None
            return ("Below is an instruction that describes a task, paired with an input that provides further context. "
                    "Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{prompt}\n\n### Input:\n{findings}\n\n### Response:")

        prompt = f"""Using the provided independent medical review information write a response JSON format. With the keys condition, treatment, approval_reason, initial_denial_reason. The value for condition should be the condition being treated, treatment being the treatment, approval_reason being why the reviewers approved it (without directly mentioning the reviwers), and initial_denial_reason being why it was denied. If the initial_denial_reason is not specified make up a reason why it might not be medically necessary."""

        return (index, [append_context_full(prompt)])


    print("Generating prompts...")
    l = imrs.apply(generate_prompts, axis=1).tolist()

    batch_size = 20

    for b in range(0, len(l), batch_size):
        print(f"Running batch {b}")
        batch = l[b: b + batch_size]

        c = 0
        start_idxs = []
        prompts = []
        for (idx, batch_prompts) in batch:
            start_idxs += [c]
            c = c + len(batch_prompts)
            prompts += batch_prompts

        try:
            print(f"Computing {len(prompts)} prompts :) {prompts}")
            results = list(map(extract_text, instruct_pipeline(prompts)))
        except Exception as e:
            print(f"Error with {e}")
            break

        print(f"Got back {len(results)}")
        ci = 0

        for (idx, batch_prompts) in batch:
            start = start_idxs[ci]
            ci = ci + 1
            i = 0
            local_results = results[start: start + len(batch_prompts)]
            for r in local_results:
                if r is None:
                    continue
                i = i + 1
                if not check_for_bad_rejection(r):
                    print(f"Writing out to {idx}MAGIC{i}_json.txt")
                    with open(join(gen_loc, f"{idx}MAGIC{i}_json.txt"), "w") as f:
                        f.write(r)
                else:
                    print(f"Skipping, found bad data in {r}")



def work_with_biogpt():
    instruct_pipeline = pipeline(model="microsoft/BioGPT-Large-PubMedQA", max_new_tokens=200)


    def generate_biogpt_hacks(imr):
        treatment = get_treatment_from_imr(imr)
        diagnosis = imr["DiagnosisSubCategory"] or imr["DiagnosisCategory"]
        findings = imr["Findings"]
        index = imr["ReferenceID"]
        if treatment is None:
            return None
        if diagnosis is not None:
            return [(index, f"{treatment} is medically necessary for {diagnosis} because")]
        else:
            return [(index, f"{treatment} is medically necessary because")]

    def write_result(res):
        idx = res[0]
        reason = extract_text(res[1])
        print(res)
        with open(join(gen_loc, f"{idx}MAGICB_appeal.txt"), "w") as o:
            o.write("""Dear [InsuranceCompany];

I am writing you to appeal claim [CLAIMNUMBER]. I believe that it is medically necessary.""")
            o.write(reason)
            o.write("\n")
            o.write("Sincerely,\n[YOURNAME]")

    batch_size = 200

    for i in range(0, len(imrs), batch_size):
        print(f"looping on batch {i}")
        batch = imrs[i: i + batch_size]
        batch_prompts = list(filter(
            not_none,
            batch.apply(generate_biogpt_hacks, axis=1).tolist()))
        idxs = map(lambda r: r[0], batch)
        qs = map(lambda r: r[1], batch)
        transformed = instruct_pipeline(list(qs))
        joined = zip(idxs, transformed)
        for res in joined:
            write_result(res)


print("Generative:")
work_with_generative()
print("biogpt:")
work_with_biogpt()
