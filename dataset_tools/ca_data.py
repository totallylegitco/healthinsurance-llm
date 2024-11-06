#!/usr/bin/python
import random
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from os.path import join
import pandas
import torch
import argparse
import re
from .utils import *
import multiprocessing
from .ca_data_utils import *


gen_loc = "generated-llm-data"

parser = argparse.ArgumentParser(
    prog="CA Data Generator", description="Generate data", epilog="Magic"
)

parser.add_argument("--small-gpu", action="store_true")

args = parser.parse_args()

def load_data(path: str): pandas.DataFrame:
    imr = pandas.read_csv(
        path,
        usecols=relevant_columns,
        dtype=str,
    )

    filtered_imr = imr[imr["Determination"].str.contains("Overturned")]
    return filtered_imr


imrs = load_data(imr_input_path)


def work_with_generative_remote():
    # Note: when adding models make sure to add to the end of the list so that
    # we apply the new model to the old records.

    print("Generating prompts...")
    l = imrs.apply(generate_prompts, axis=1).tolist()

    for r in l:
        idx = r[0]
        for (m, model_index) in models:
            # For the first model we don't add an idex but subsequent ones we do.
            mistr = ""
            if model_index > 0:
                mistr = f"{model_index}-"
            model_index = model_index + 1
            print(r[2])
            for (k, v) in r[2].items():
                target_file = join(gen_loc, f"{idx}EXTRACTED{k}.txt")
                if not os.path.exists(target_file):
                    with open(target_file, "w") as f:
                        f.write(v)
            if m is None:
                print("Skipping disabled model.")
                continue
            results = {}
            for response_type in prompt_order:
                if response_type not in r[1]:
                    continue
                i = 0
                for v in r[1][response_type]:
                    target_file = join(
                        gen_loc, f"{idx}MAGIC{mistr}{i}{response_type}.txt")
                    i = i + 1
#                    if not os.path.exists(target_file) and (response_type != "appeal" or not check_for_bad_file(response_type, response)):
                    if not os.path.exists(target_file) or check_for_bad_file(response_type, target_file):
                        # Sub in previous responses
                        if "#" in v:
                            for prev in results.keys():
                                v = v.replace(f"#{prev}#", results[prev])
                        response = make_request(m, v)
                        # If we find an invalid url ask the model to go again
                        if check_for_invalid_urls(response):
                            bad_urls = list_invalid_urls(response)
                            error = f"You referenced some invalid urls {bad_urls}."
                            response = make_request(m, v, response, error)
                        bad_results = check_for_bad_and_return_result(response_type, response)
                        if len(bad_results) > 0:
                            error = f"You referenced {bad_results} don't do that your writing as if it was before the reviewers looked & don't make up any references which are not in the input."
                            response = make_request(m, v, response, error)
                        results[response_type] = response
                        if not check_for_bad(response_type, response):
                            print(f"Writing out to {target_file}")
                            with open(target_file, "w") as f:
                                f.write(response)
                        else:
                            print(f"Bad response {response} skipping for now")
                    else:
                        with open(target_file) as x: results[response_type] = x.read()
                        print(
                            f"We already good data for {target_file} skipping..")


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
        batch = imrs[i: i + batch_size]
        batch_prompts = list(
            filter(not_none, batch.apply(
                generate_biogpt_hacks, axis=1).tolist())
        )
        idxs = map(lambda r: r[0], batch)
        qs = map(lambda r: r[1], batch)
        transformed = instruct_pipeline(list(qs))
        joined = zip(idxs, transformed)
        for res in joined:
            write_result(res)


print("Generative:")
work_with_generative_remote()
# work_with_generative_local()
print("biogpt:")
work_with_biogpt()
