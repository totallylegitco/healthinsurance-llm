import pandas as pd
import nltk
from nltk.corpus import stopwords
import json
from os import listdir
from os.path import isfile, join
import re
from .utils import *
from PyPDF2 import PdfReader

magic_re = re.compile(r".*/(.*?)(MAGIC[0-9B]*|FARTS[0-9]*|)_?(appeal|rejection|json).txt")

with open('header.txt') as x: header = x.read()
with open('alt_header.txt') as x: alt_header = x.read()
with open('mt_header.txt') as x: mt_header = x.read()

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
    if groups is not None:
        return groups.group(1)
    else:
        return None

def insert_into_case_dict(filename):
    case = file_name_to_case(filename)
    if case is not None:
        lt = letter_type(filename)
        if case not in cases:
            cases[case] = {"appeal": [], "rejection": [], "json": []}
        cases[case][lt] += [filename]

for f in filter(check_record, data_files):
    insert_into_case_dict(f)

# This is going to be an explosion! But intentional.
recommend_regex = re.compile(r"recommends* ([^.]+)\.", re.IGNORECASE)
short_recommendations = set()
alpaca = open("out/train_alpaca.jsonl", "w")
alpaca.write("[")
alpaca_smaller = open("out/train_alpaca_smaller.jsonl", "w")
alpaca_smaller.write("[")


first_alpaca = True
first_alpaca_small = True

so = open("out/train_smaller.jsonl", "w")
o = open("out/train.jsonl", "w")

def format_dolly(instruction, result, context):
    record = json.dumps({
        "instruction": instruction,
        "context": context,
        "response": result,
        "category": "open_qa"})
    record.replace("\n", " ")
    return record + "\n"

def write(instruction, result, context=""):
    write_dolly(instruction, result, context=context)
    write_alpaca(instruction, result, context=context)

def write_small(instruction, result, context=""):
    write_dolly_small(instruction, result, context=context)
    write_alpaca_small(instruction, result, context=context)
    
def write_dolly_small(instruction, result, context=""):
    so.write(format_dolly(instruction, result, context))

def write_dolly(instruction, result, context=""):
    o.write(format_dolly(instruction, result, context))

def format_alpaca(instruction, result, context=""):
    alpaca_record = json.dumps({
        "instruction": instruction,
        "input": context,
        "output": result})
    alpaca_record.replace("\n", " ")
    return alpaca_record + "\n"

def write_alpaca(instruction, result, context=""):
    global first_alpaca
    if first_alpaca:
        first_alpaca = False
    else:
        alpaca.write(",")
    alpaca_record = format_alpaca(instruction, result, context)
    alpaca.write(alpaca_record)
    
    
def write_alpaca_small(instruction, result, context=""):
    global first_alpaca_small
    if first_alpaca_small:
        first_alpaca_small = False
    else:
        alpaca_smaller.write(",")
    alpaca_record = format_alpaca(instruction, result, context)
    alpaca_smaller.write(alpaca_record)    

def process_pdf(pdf):
    print(f"Loading {pdf}")
    reader = PdfReader(pdf)
    c = 0
    recs = set()
    for page in reader.pages:
        t = page.extract_text().replace("-\n", "").replace("\n", "").replace(
            "\u201c", '"').replace("\u201d", '"')
        results = set()
        for match in re.finditer(recommend_regex, t):
            result = match.group(1)
            results.add(result)
        for result in results:
            # TODO: remove stopwords w/nltk
            short = result.lower().replace("the", "").replace(" ", "")
            if short not in short_recommendations:
                short_recommendations.add(short)
                instruction = f"What is one of the recommendations in {pdf}?"
                write_alpaca(instruction, result)
                write_dolly(instruction, result)
                recs.add(result)

    prompt = f"What are the recommendations in {pdf}"
    expect = "-\n".join(recs)
    write(prompt, expect)
    write_small(prompt, expect)


for pdf in pdfs:
    process_pdf(pdf)


def write_chemo_drug_records():
    parsed_chemo = pd.read_csv("./data_sources/parsed_chemo_drugs.csv")
    pcl = parsed_chemo.tolist()

    for r in pcl:
        write(r["question"], result=r["answer"], context="")


def write_mt_sample_contexts():
    mt = pd.read_csv("./data_sources/mtsamples2.csv")
    mtl = mt.tolist()

    for r in mtl:
        write(mt_header, r["transcription"], context=r["description"])


def write_10k():
    ic = pd.read_csv("./data_sources/ic10k.csv")
    icl = ic.tolist()

    for r in icl:
        write(r["input"], r["answer_icliniq"])
        
for (case_key, case) in cases.items():
    try:
        for j in case["json"]:
            write(
                "Why should the the provided treatment be covered.",
                j["approval_reason"],
                j["treatment"])
            write(
                "Why should the provided denial be overturned?",
                j["approval_reason"],
                j["initial_denial_reason"])
        for r in case["rejection"]:
            rejection = load_record(r)
            if r is None or r == "null":
                continue
            for a in case["appeal"]:
                appeal = load_record(a)
                if (a is None or a == "null" or a == "" or
                    len(a) < 10):
                    continue
                write(header, appeal, rejection)
                if "MAGIC" not in r:
                    write_small(header, appeal, rejection)

    except Exception as e:
        print(f"Exception {e} while processing case {case}")

alpaca.write("]")
alpaca_smaller.write("]")
