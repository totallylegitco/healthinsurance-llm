#!/usr/bin/python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline)
from os import listdir
from os.path import join
import pandas
import torch
import argparse
import re

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
    r"""\s*(The|An|A)?\s*(parent|father|mother|patient|enrollee|member)\s*[^.]*(requested|required|asked|requires|reimbursement|coverage|requesting)\s*[^.]*(of|for|medication|reimbursement|coverage|services)\s+(\d*\w+.+?)\.""",
    re.IGNORECASE)
alt_treatment_regex = re.compile(
    r"""At issue\s*(in this case|)\s*(is|)\s*(whether|if)\s+(\d*\w+.+?) (is|were|was) medically (necessary|indicated)""",
    re.IGNORECASE)
more_alt_treatment_regex = re.compile(
    r"""the requested (medication|treatment|service|procedure)\s+(\d*\w+.+?) (is|were|was) (likely to be|medically necessary|medically indicated)""",
    re.IGNORECASE)

even_more_alt_treatment_regex = re.compile(
    r"""(Therefore|Thus|As such),[^.]*?\s+(an|a|the|that) (\w+[^.]+?) (is|were|was|should be) (medically necessary|medically indicated|likely to be|authorized)""",
    re.IGNORECASE)

perscribed_regex = re.compile(
    r"""patients provider has prescribed the medication\s+([^.]+?).""", re.IGNORECASE)

wishes_to_regex = re.compile(r"""(wishes|desires|like) to (undergo|take)\s+([^.]+?).""", re.IGNORECASE)

health_plan_not_necessary_regex = re.compile(r"""The (Health Plan|Plan|Insurance Company) (determined the|determined|indicates) (.+?) (is|was|were) not""", re.IGNORECASE)

almost_sketchy_regex = re.compile(r"""treatment[^.]*with\s+([^.]+?) (is|were|was)""", re.IGNORECASE)

sketchy_regex = re.compile(r"""(requested|required|asked|requires|reimbursement|coverage|request|requesting)\s*[^.]*(for|medication|reimbursement|coverage|of)\s+(\d*\w+.+?)\.""",
    re.IGNORECASE)

def get_treatment_from_imr(imr):
    findings = imr["Findings"]
    result = treatment_regex.search(findings)
    alt_result = alt_treatment_regex.search(findings)
    more_alt_result = more_alt_treatment_regex.search(findings)
    even_more_alt_result = even_more_alt_treatment_regex.search(findings)
    perscribed_result = perscribed_regex.search(findings)
    wishes_to_result = wishes_to_regex.search(findings)
    almost_sketchy_result = almost_sketchy_regex.search(findings)
    health_plan_not_necessary_result = health_plan_not_necessary_regex.search(findings)
    sketchy_result = sketchy_regex.search(findings)
    if result is not None:
        return result.group(5)
    elif alt_result is not None:
        return alt_result.group(4)
    elif more_alt_result is not None:
        return more_alt_result.group(2)
    elif even_more_alt_result is not None:
        return even_more_alt_result.group(3)
    elif perscribed_result is not None:
        return perscribed_result.group(1)
    elif wishes_to_result is not None:
        return wishes_to_result.group(3)
    elif health_plan_not_necessary_result is not None:
        return health_plan_not_necessary_result.group(3)
    elif almost_sketchy_result is not None:
        return almost_sketchy_result.group(1)
    elif sketchy_result is not None:
        return sketchy_result.group(3)
    else:
        print(f"No match in {findings}")
    return imr["TreatmentSubCategory"] or imr["TreatmentCategory"]


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
                    device_map="auto")
            else:
                instruct_pipeline = pipeline(
                    model=model,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto")
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
        diagnosis = imr["DiagnosisSubCategory"] or imr["DiagnosisCategory"]
        findings = imr["Findings"].strip("\n")
        grounds = imr["Type"]
        index = imr["ReferenceID"]

        def append_context(prompt):
            if "dolly" in m:
                return append_context_dolly(prompt)
            else:
                return append_context_alpasta(prompt)
                
        def append_context_dolly(prompt):
            return f"""{prompt}

Input:
On review the following was found {findings}"""
        
        def append_context_alpasta(prompt):
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Input:
{findings}

### Response:
"""
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

        return (index,
                list(map(append_context, rejection_prompts)),
                list(map(append_context, appeal_prompts)))

    def cleanup_appeal(text):
        if text is None:
            return None
        sentences = text.split(".")
        less_sketchy = ".".join(filter(sketchy_sentence_filter, sentences))
        if len(less_sketchy) < 40:
            return None
        if (not "Dear" in less_sketchy) and not ("To Whom" in less_sketchy):
            less_sketchy = f"Dear [INSURANCECOMPANY];\n{less_sketchy}"
        return less_sketchy

    def cleanup_rejection(text):
        if text is None:
            return None
        if not "denied" in text:
            text = f"{text}. Your request is denied."
        if not "[MEMBER]" in text:
            text = f"Dear [MEMBER]; {text}."
        if not "appeal" in text:
            text = f"{text}. You have the right to appeal this decision."
        def mark_unnecessary(match_objc):
            return f"{match_obj.group(1)} not medically {match_object.group(2)}"
        text = re.sub(r"(is|are|were)\s*medically\s*(necessary|required)", mark_unnecessary)
        return text



    l = imrs.apply(generate_prompts, axis=1).tolist()

    batch_size = 5

    for b in range(0, len(l), batch_size):
        batch = l[b: b + batch_size]
        prompts = []

        c = 0
        start_idxs = []
        for (idx, rejection_prompts, appeal_prompts) in batch:
            combined = rejection_prompts + appeal_prompts
            prompts += combined
            start_idxs += [c]
            c = c + len(combined)

        ci = 0
        for (idx, rejection_prompts, appeal_prompts) in batch:
            start = start_idxs[ci]
            ci = ci + 1
            results = list(map(extract_text, instruct_pipeline(prompts)))
            rejections = map(cleanup_rejection, results[start:len(rejection_prompts)])
            appeals = map(cleanup_appeal, results[len(rejection_prompts):len(appeal_prompts)])
            i = 0
            for r in rejections:
                if r is None:
                    continue
                i = i + 1
                with open(join(gen_loc, f"{idx}MAGIC{i}_rejection.txt"), "w") as f:
                    f.write(r)
            i = 0
            for a in appeals:
                if a is None:
                    continue
                i = i + 1
                with open(join(gen_loc, f"{idx}MAGIC{i}_appeal.txt"), "w") as f:
                    f.write(a)




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
