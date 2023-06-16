import json
from os import listdir
from os.path import isfile, join
import re
from .utils import *
from PyPDF2 import PdfReader

magic_re = re.compile(r".*/(.*?)(MAGIC[0-9B]*|)_?(appeal|rejection).txt")

with open('header.txt') as x: header = x.read()

category = "creative_writing"

raw_dataset = "combined-llm-data"

listed = map(lambda f: join(raw_dataset, f), listdir(raw_dataset))

pdf_sources = "data_sources"

raw_listed = map(lambda f: join(pdf_sources, f), listdir(pdf_sources))

pdfs = [f for f in raw_listed if (isfile(f) and f.endswith(".pdf"))]

data_files = [f for f in listed if (isfile(f))]

# Make a dctionary of the cases + files associated with them.

cases = {}

def file_name_to_case(filename):
    groups = magic_re.search(filename)
    return groups.group(1)

def insert_into_case_dict(filename):
    case = file_name_to_case(filename)
    lt = letter_type(filename)
    if case not in cases:
        cases[case] = {"appeal": [], "rejection": []}
    cases[case][lt] += [filename]

for f in filter(check_record, data_files):
    insert_into_case_dict(f)

# This is going to be an explosion! But intentional.
recommend_regex = re.compile(r"recommends* ([^.]+)\.", re.IGNORECASE)
alpaca = open("out/train_alpaca.jsonl", "w")
alpaca.write("[")

first = True

with open("out/train.jsonl", "w") as o:
    with open("out_oa/train.jsonl", "w") as oa:
        def process_pdf(pdf):
            print(f"Loading {pdf}")
            reader = PdfReader(pdf)
            c = 0
            for page in reader.pages:
                t = page.extract_text().replace("-\n", "").replace("\n", "").replace(
                    "\u201c", '"').replace("\u201d", '"')
                results = set()
                for match in re.finditer(recommend_regex, t):
                    result = match.group(1)
                    results.add(result)
                for result in results:
                    instruction = f"What is one of the recommendations in {pdf}?"
                    record = json.dumps({
                        "instruction": instruction,
                        "context": "",
                        "response": result,
                        "category": "open_qa"})
                    record.replace("\n", "")
                    o.write(record)
                    o.write("\n")
                    alpaca_record = json.dumps({
                        "instruction": instruction,
                        "input": "",
                        "output": result})
                    alpaca_record.replace("\n", "")
                    global first
                    if first:
                        first = False
                    else:
                        alpaca.write(",")
                    alpaca.write(alpaca_record)
                    alpaca.write("\n")
                    simple_text = f"### Human {instruction} ### Assistant {result}"
                    simple_record = json.dumps({"text": simple_text})
                    simple_record.replace("\n", "")
                    oa.write(simple_record)


        for pdf in pdfs:
            process_pdf(pdf)

        for (case_key, case) in cases.items():
            try:
                for r in case["rejection"]:
                    rejection = load_record(r)
                    if r is None or r == "null":
                        continue
                    for a in case["appeal"]:
                        appeal = load_record(a)
                        if (a is None or a == "null" or a == "" or
                            len(a) < 10):
                            continue
                        prompt = f"{header}{rejection}"
                        record = json.dumps({
                            "instruction": prompt,
                            "context": "",
                            "response": appeal,
                            "category": category})
                        record.replace("\n", "")
                        o.write(record)
                        o.write("\n")
                        alpaca_record = json.dumps({
                            "instruction": prompt,
                            "input": "",
                            "output": appeal})
                        alpaca_record.replace("\n", "")
                        alpaca.write(",")
                        alpaca.write(alpaca_record)
                        alpaca.write("\n")
                        simple_text = f"### Human {prompt} ### Assistant {appeal}"
                        simple_record = json.dumps({
                            "text": simple_text})
                        simple_record.replace("\n", "")
                        oa.write(simple_record)

            except Exception as e:
                print(f"Exception {e} while processing case {case}")

alpaca.write("]")
