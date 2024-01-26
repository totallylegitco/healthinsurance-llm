#!/usr/bin/python
import random
import backoff
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
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
    prog="CA Data Generator", description="Generate data", epilog="Magic"
)

parser.add_argument("--small-gpu", action="store_true")

args = parser.parse_args()

treatment_regex = re.compile(
    r"""\s*(The|An|A)?\s*(parent|father|mother|patient|enrollee|member|provider)\s*[^.]*(requested|required|asked|requires|reimbursement|coverage|requesting|has)\s*[^.]*(of|for|medication|reimbursement|coverage|services)\s+(?P<treatment>\d*\w+.+?)\.""",
    re.IGNORECASE,
)
alt_treatment_regex = re.compile(
    r"""At issue\s*(in this case|)\s*(is|)\s*(whether|if)\s+(?P<treatment>\d*\w+.+?) (is|were|was) medically (necessary|indicated)""",
    re.IGNORECASE,
)
more_alt_treatment_regex = re.compile(
    r"""the requested (medication|treatment|service|procedure)\s+(?P<treatment>\d*\w+.+?) (is|were|was) (likely to be|medically necessary|medically indicated)""",
    re.IGNORECASE,
)

even_more_alt_treatment_regex = re.compile(
    r"""(Therefore|Thus|As such),[^.]*?\s+(an|a|the|that) (?P<treatment>\w+[^.]+?) (is|were|was|should be) (medically necessary|medically indicated|likely to be|authorized)""",
    re.IGNORECASE,
)

perscribed_regex = re.compile(
    r"""patients provider has prescribed the medication\s+(?P<treatment>[^.]+?).""",
    re.IGNORECASE,
)

wishes_to_regex = re.compile(
    r"""(wishes|desires|like) to (undergo|take)\s+(?P<treatment>[^.]+?).""",
    re.IGNORECASE,
)

health_plan_not_necessary_regex = re.compile(
    r"""The (Health Plan|Plan|Insurance Company) (determined the|determined|indicates) (?P<treatment>.+?) (is|was|were) not""",
    re.IGNORECASE,
)

almost_sketchy_regex = re.compile(
    r"""treatment[^.]*with\s+(?P<treatment>[^.]+?) (is|were|was)""", re.IGNORECASE
)

sketchy_regex = re.compile(
    r"""(requested|required|asked|requires|reimbursement|coverage|request|requesting)\s*[^.]*(for|medication|reimbursement|coverage|of)\s+(?P<treatment>\d*\w+.+?)\.""",
    re.IGNORECASE,
)

seeking_regex = re.compile(
    r"""is seeking (?P<treatement>) for (?P<diagnosis>[^.]+)""", re.IGNORECASE
)
admitted_regex = re.compile(
    r"""admitted to the hospital for (?P<diagnosis>[^.]+)""", re.IGNORECASE
)
recommended_regex = re.compile(
    r"""physicians recommended (?P<treatment>[^.]+)""", re.IGNORECASE
)

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
    recommended_regex,
]

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
with open("bad_appeal_strings.txt") as f:
    bad_appeal_strings = f.read().split("\n")


def load_data(path):
    imr = pandas.read_csv(
        path,
        usecols=[
            "Determination",
            "TreatmentCategory",
            "TreatmentSubCategory",
            "DiagnosisCategory",
            "DiagnosisSubCategory",
            "Type",
            "Findings",
            "ReferenceID",
        ],
        dtype=str,
    )

    filtered_imr = imr[imr["Determination"].str.contains("Overturned")]
    return filtered_imr


imrs = load_data(
    "./data_sources/ca-independent-medical-review-imr-determinations-trends-utf8.csv"
)

def generate_prompts_instruct(imr):
    def format_for_model(x):
        return f"<s>[INST]{x}[/INST]"

    return generate_prompts(imr, format_for_model=format_for_model)

def generate_prompts(imr, format_for_model = lambda x: x):
    determination = imr["Determination"]
    treatment = get_treatment_from_imr(imr)
    diagnosis = get_diagnosis_from_imr(imr)
    findings = imr["Findings"].strip("\n")
    grounds = imr["Type"]
    index = imr["ReferenceID"]
    treatment_extra = ""
    if treatment is not None:
        treatment_extra = " and treatment {treatment}"

    prompts = {
        "denial": [
            format_for_model(
                f"""The independent medical review findings were {findings} and grounds were {grounds}{treatment_extra}. Use this information to write the original insurance denial from the insurance company. Do not include any reference to the reviewers or their findings, instead focus on what the insurance company would have written denying the patients first claim. Keep in mind the denial would have been written before the independent review. Feel free to be verbose. You may wish to start your denial as a letter with \"Dear [Member];\""""),
            format_for_model(
                f"""Given the following medical reviewer findings:

{findings}
                Compose an initial rejection letter on behalf of the insurance company in response to a patient's request for medical coverage. Include specific details about the patient's case, addressing the reasons for denial without referencing any independent medical review findings. Ensure the letter is concise, professional, and clearly communicates the grounds for the denial. Focus on policy justifications, eligibility criteria, medical necessity, or any other relevant factors that would lead to the initial rejection. Omit any mention of the independent medical reviewers' assessments or findings as those happend later in the process.""")
        ],
        "appeal": [
            format_for_model(
                f"""The independent medical review findings were {findings} and grounds were {grounds}{treatment_extra}. In your response you are writing on your own on behalf (not that of a doctors office) and you do not have any credentials. Do not include any reference to the reviewers or their findings. Use this information to write the original appeal by the patient. Keep in mind the appeal would be written before the appeal. Remember you are writing for yourself, not on behalf of anyone else. If any studies or guidelines support the medical necessity include them. Feel free to be verbose and start your appeal with Dear [Insurance Company];"""),
            format_for_model(
                f"""Given the following medical reviewer findings:\n{findings}\n Do not include any information about the reviewers' findings. Instead, consider the patient's personal experience, medical history, and reasons for seeking the requested medical coverage. Craft the appeal to express the patient's perspective and emphasize their need for the requested medical intervention without referencing the independent medical review outcomes. Omit any mention of the independent medical reviewers' assessments or findings as those happend later in the process. Feel free to be verbose and write in the style of patio11 or a bureaucrat like sir humphrey appleby. Remember you are writing for yourself, not on behalf of anyone else. If any studies or guidelines support the medical necessity include them."""),
        ],
        "medically_necessary": [
            format_for_model(
                f"""Given the following medical review findings: {findings} and grounds were {grounds}{treatment_extra}. Why was the treatment considered medically necessary? Don't refer to the reviewers findings directly instead write in a general fashion. For example if the reviewers found that facial feminization surgery was needed to treat gender dysphoria based on WPATH guidelines you would write something like: Facial feminization surgery is medically necessary for gender dysphoria per the WPATH guidelines. Do not refer to the reviewers qualifications or the reviewers themselves directly. If any studies or guidelines support the medical necessity include them."""),
        ],
        "reason_for_denial": [
            format_for_model(f"""Given the following medical review findings:  {findings} and grounds were {grounds}{treatment_extra}. What excuse did the insurance company use to deny the treatment? Some common reasons are medical necessary, STEP treatment required, experimental treatments, or a procedure being considered cosmetic. These are just examples though, insurance companies can deny care for many reasons. What was the reason here?""")
        ],
        "treatment": [
            format_for_model(f"""Based on the independent review findings: {findings}. What was the treatment, procedure, therapy, or surgery denied?""")
        ]
    }

    return (index, prompts)


def work_with_generative_remote():
    backend = os.getenv("BACKEND_PROVIDER", "https://api.perplexity.ai/chat/completions")
    print(f"Using backend {backend}")

    @backoff.on_exception(
        backoff.expo, requests.exceptions.RequestException, max_time=600
    )
    def make_request(model, prompt):
        # Perplexity is an interesting backend for personal use.
        # The inference costs are a little high though for full training data
        # creation so look for whoever is cheapest when running in prod.
        # deepinfra was cheap when working on this last. Always check TOS
        # See https://artificialanalysis.ai/
        url = backend

        token = os.getenv("SECRET_BACKEND_TOKEN", os.getenv("PERPLEXITY_API"))
        if token is None:
            print("Error no Token provided for perplexity.")

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {token}",
        }

        time.sleep(random.randint(0, 15))
        print(f"Making request for {model} and {prompt}")
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        response_text = response.json()["choices"][0]["message"]["content"]
        print(f"Promxspt: {prompt}\nResponse text: {response_text}")
        return response_text

    # Note: when adding models make sure to add to the end of the list so that
    # we apply the new model to the old records.
    models = [
        #"mistral-7b-instruct",
        #"openhermes-2-mistral-7b",
        ("mistralai/Mixtral-8x7B-Instruct-v0.1", 3)]

    print("Generating prompts...")
    l = imrs.apply(generate_prompts, axis=1).tolist()

    for r in l:
        print(r[1])
        for (m, model_index) in models:
            # For the first model we don't add an idex but subsequent ones we do.
            mistr = ""
            if model_index > 0:
                mistr = f"{model_index}-"
            model_index = model_index + 1
            if m is None:
                print("Skipping disabled model.")
                continue
            for response_type in r[1].keys():
                i = 0
                for v in r[1][response_type]:
                    idx = r[0]
                    target_file = join(gen_loc, f"{idx}MAGIC{mistr}{i}{response_type}.txt")
                    i = i + 1
                    if not os.path.exists(target_file) or not check_for_bad_file(response_type, target_file):
                        response = make_request(m, v)
                        if not check_for_bad(response_type, response):
                            if not check_for_bad(response_type, response):
                                print(f"Writing out to {target_file}")
                                with open(target_file, "w") as f:
                                    f.write(response)
                            else:
                                print(
                                    f"Skipping, found bad data in {r} for rt {response_type}"
                                )
                    else:
                        print(f"We already have good data for {target_file} skipping..")


def work_with_generative_local():
    # Load the model to do our magic

    candidate_models = [
        #        "ausboss/llama-30b-supercot",
        #        "CalderaAI/30B-Lazarus",
        #        "tiiuae/falcon-40b-instruct",
        ("mistralai/Mixtral-8x7B-Instruct-v0.1", 2),
#        "teknium/OpenHermes-2-Mistral-7B",
#        "TheBloke/OpenHermes-2-Mistral-7B-GPTQ",
#        ("mistralai/Mistral-7B-v0.1", 0)
#        "databricks/dolly-v2-12b",
#        "databricks/dolly-v2-7b",
#        "databricks/dolly-v2-3b",
    ]

    # We load the model first to make sure we can actually do magic
    instruct_pipeline = None

    for (model, model_index) in candidate_models:
        mistr = ""
        if model_index > 0:
            mistr = f"{model_index}-"
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
                    model_kwargs={"load_in_8bit": True},
                    device_map="auto",
                    max_new_tokens=1124,
                )
            else:
                instruct_pipeline = pipeline(
                    model=model,
                    tokenizer=tokenizer,
                    eos_token_id=tokenizer.eos_token_id,
                    top_k=10,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto",
                    max_new_tokens=1124,
                )
            break
        except Exception as e:
            print(f"Error {e} loading {model}")

    if instruct_pipeline is None:
        raise Exception("Could not load any model")

    print("Generating prompts...")
    l = imrs.apply(generate_prompts_instruct, axis=1).tolist()
    batch_size = 20

    for b in range(0, len(l), batch_size):
        print(f"Running batch {b}")
        batch = l[b : b + batch_size]

        for k in batch[0].keys():
            mybatch = list(map(lambda x: x[k], batch))
            c = 0
            start_idxs = []
            prompts = []
            for idx, batch_prompts in mybatch:
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

            for idx, batch_prompts in mybatch:
                start = start_idxs[ci]
                ci = ci + 1
                i = 3
                local_results = results[start : start + len(batch_prompts)]
                for r in local_results:
                    if r is None:
                        continue
                    i = i + 1
                    if not check_for_bad(response_type, r):
                        print(f"Writing out to {idx}MAGIC{mistr}{i}{response_type}.txt")
                        with open(
                            join(gen_loc, f"{idx}MAGIC{mistr}{i}{response_type}.txt"), "w"
                        ) as f:
                            f.write(r)
                    else:
                        print(f"Skipping, found bad data in {r} for rt {response_type}")


def work_with_biogpt():
    instruct_pipeline = pipeline(
        model="microsoft/BioGPT-Large-PubMedQA", max_new_tokens=200
    )

    def generate_biogpt_hacks(imr):
        treatment = get_treatment_from_imr(imr)
        diagnosis = imr["DiagnosisSubCategory"] or imr["DiagnosisCategory"]
        findings = imr["Findings"]
        index = imr["ReferenceID"]
        if treatment is None:
            return None
        if diagnosis is not None:
            return [
                (index, f"{treatment} is medically necessary for {diagnosis} because")
            ]
        else:
            return [(index, f"{treatment} is medically necessary because")]

    def write_result(res):
        idx = res[0]
        reason = extract_text(res[1])
        print(res)
        with open(join(gen_loc, f"{idx}MAGICB_appeal.txt"), "w") as o:
            o.write(
                """Dear [InsuranceCompany];

I am writing you to appeal claim [CLAIMNUMBER]. I believe that it is medically necessary."""
            )
            o.write(reason)
            o.write("\n")
            o.write("Sincerely,\n[YOURNAME]")

    batch_size = 200

    for i in range(0, len(imrs), batch_size):
        print(f"looping on batch {i}")
        batch = imrs[i : i + batch_size]
        batch_prompts = list(
            filter(not_none, batch.apply(generate_biogpt_hacks, axis=1).tolist())
        )
        idxs = map(lambda r: r[0], batch)
        qs = map(lambda r: r[1], batch)
        transformed = instruct_pipeline(list(qs))
        joined = zip(idxs, transformed)
        for res in joined:
            write_result(res)


print("Generative:")
work_with_generative_remote()
#work_with_generative_local()
print("biogpt:")
work_with_biogpt()
