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

pool = multiprocessing.Pool()

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
        "tiiuae/falcon-40b-instruct",
        "databricks/dolly-v2-12b",
        "databricks/dolly-v2-7b",
        "databricks/dolly-v2-3b",
    ]

    instruct_pipeline = None

    m = None

    for model in candidate_models:
        try:
            m = model
            print(f"Loading {m}\n")
            if args.small_gpu:
                instruct_pipeline = pipeline(
                    model=model,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    model_kwargs={'load_in_8bit': True},
                    device_map="auto",
                    max_new_tokens=512
                )
            else:
                instruct_pipeline = pipeline(
                    model=model,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto",
                    max_new_tokens=512)
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
        determination = imr["Determination"]
        treatment = get_treatment_from_imr(imr)
        diagnosis = get_diagnosis_from_imr(imr)
        findings = imr["Findings"].strip("\n")
        grounds = imr["Type"]
        index = imr["ReferenceID"]

        def append_context(prompt):
            return [{"instruction": prompt, "input": determination}]

        rejection_prompts = [
            f"What was the reason that {treatment} was originally denied in the provided determination.",
            f"Write a health insurance denial for {treatment} for diagnosis {diagnosis} on the grounds of {grounds} that could have resulted in the provided determination.",
            f"Deny coverage for {treatment} for {diagnosis} that could have resulted in the provided determination.",
            f"Write a denial for {treatment} that could have resulted in the provided determination.",
        ]
        appeal_prompts = [
            f"Write an appeal for {treatment} that could have resulted in the provided determination.",
            f"Write an appeal for {treatment} for {diagnosis} that could have resulted in the provided determination.",
        ]

        return (index,
                list(flatten(pool.map(append_context, rejection_prompts))),
                list(flatten(pool.map(append_context, appeal_prompts))))

    def training_cleanup_appeal(text):
        if text is None:
            return None
        sentences = text.split(".")
        less_sketchy = ".".join(filter(sketchy_sentence_filter, sentences))
        if len(less_sketchy) < 40:
            return None
        if (not "Dear" in less_sketchy) and not ("To Whom" in less_sketchy):
            less_sketchy = f"Dear [INSURANCECOMPANY];\n{less_sketchy}"
        return cleanup_appeal(less_sketchy)

    def training_cleanup_rejection(text):
        if text is None:
            return None
        if not "denied" in text and not "no additional treatment" in text:
            text = f"{text}. Your request is denied."
        if not "[MEMBER]" in text:
            text = f"Dear [MEMBER]; {text}."
    
        def mark_unnecessary(match):
            return f"{match.group(1)} not medically {match.group(2)}"
        text = re.sub(r"(is|are|were|be)\s*medically\s*(necessary|required)", mark_unnecessary, text)
        return cleanup_rejection(text)



    print("Generating prompts...")
    prompt_batch_size = 10
    l = imrs.apply(generate_prompts, axis=1).tolist()

    batch_size = 10

    for b in range(0, len(l), batch_size):
        print(f"Running batch {b}")
        batch = l[b: b + batch_size]

        c = 0
        start_idxs = []
        prompts = []
        for (idx, rejection_prompts, appeal_prompts) in batch:
            combined = rejection_prompts + appeal_prompts
            start_idxs += [c]
            c = c + len(combined)
            prompts += combined

        try:
            print(f"Computing {len(prompts)} prompts :)")
            results = list(map(extract_text, instruct_pipeline(prompts)))
        except Exception as e:
            print(f"Error with {e}")
            break

        print(f"Got back {len(results)}")
        ci = 0

        for (idx, rejection_prompts, appeal_prompts) in batch:
            start = start_idxs[ci]
            ci = ci + 1
            rejections = pool.map(
                training_cleanup_rejection,
                results[start:
                        start + len(rejection_prompts)])
            appeals = pool.map(
                training_cleanup_appeal,
                results[start + len(rejection_prompts):
                        start + len(rejection_prompts) + len(appeal_prompts)])
            i = 0
            def handle_rejection(r):
                if r is None:
                    return
                i = i + 1
                if not check_for_bad_rejection(r):
                    print(f"Writing out to {idx}MAGIC{i}_rejection.txt")
                    with open(join(gen_loc, f"{idx}MAGIC{i}_rejection.txt"), "w") as f:
                        f.write(r)
                else:
                    print(f"Skipping, found bad data in {r}")
            i = 0
            def handle_appeal(a):
                if a is None:
                    return
                i = i + 1
                if not check_for_bad_rejection(r):
                    with open(join(gen_loc, f"{idx}MAGIC{i}_appeal.txt"), "w") as f:
                        f.write(a)
                else:
                    print(f"Skipping, found bad data in {a}")
            pool.map(handle_appeal, appeals)
            pool.map(handle_rejection, rejections)




def work_with_biogpt():
    instruct_pipeline = pipeline(model="microsoft/BioGPT-Large-PubMedQA", max_new_tokens=100)


    def generate_biogpt_hacks(imr):
        treatment = get_treatment_from_imr(imr)
        diagnosis = imr["DiagnosisSubCategory"] or imr["DiagnosisCategory"]
        findings = imr["Findings"]
        index = imr["ReferenceID"]
        return (index, f"{treatment} is medically necessary for {diagnosis} because")

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

    l = imrs.apply(generate_biogpt_hacks, axis=1).tolist()

    batch_size = 200

    for i in range(0, len(l), batch_size):
        batch = l[i: i + batch_size]
        idxs = map(lambda r: r[0], batch)
        qs = map(lambda r: r[1], batch)
        transformed = instruct_pipeline(list(qs))
        joined = zip(idxs, transformed)
        for res in joined:
            write_result(res)


work_with_generative()
work_with_biogpt()
