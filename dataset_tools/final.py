import pandas as pd
import nltk
from nltk.corpus import stopwords
import json
from os import listdir
from os.path import isfile, join
import re
from .utils import *
from PyPDF2 import PdfReader

max_answer_len = 20480

with open("header.txt") as x:
    header = x.read()
with open("alt_header.txt") as x:
    alt_header = x.read()
with open("mt_header.txt") as x:
    mt_header = x.read()

category = "creative_writing"

raw_dataset = "combined-llm-data"

listed = map(lambda f: join(raw_dataset, f), listdir(raw_dataset))

pdf_sources = "data_sources"

raw_listed = map(lambda f: join(pdf_sources, f), listdir(pdf_sources))

pdfs = [f for f in raw_listed if (isfile(f) and f.endswith(".pdf"))]

data_files = [f for f in listed if (isfile(f))]

# Make a dctionary of the cases + files associated with them.

cases = {}


def insert_into_case_dict(filename):
    case = file_name_to_case(filename)
    if case is not None:
        lt = letter_type(filename)
        print(f"Adding {case} of type {lt}")
        if case not in cases:
            cases[case] = {"appeal": [], "rejection": [], "json": []}
        cases[case][lt] += [filename]


for f in filter(check_record, data_files):
    insert_into_case_dict(f)

# This is going to be an explosion! But intentional.
recommend_regex = re.compile(r"recommends* ([^.]+)\.", re.IGNORECASE)
short_recommendations = set()


so = open("out/train_smaller.jsonl", "w")
o = open("out/train.jsonl", "w")


def format(system, instruction, result, context=""):
    # Handle system
    if system is None:
        system = ""
    if len(system) > 0:
        if "<<SYS>>" not in system:
            system = f"<<SYS>>{system}<</SYS>>"
    alpaca_record = json.dumps(
        {
            "instruction": "[INST] " + system + instruction + " " + context + "[/INST]",
            "output": result[0:max_answer_len],
        }
    )
    alpaca_record.replace("\n", " ")
    return alpaca_record + "\n"


def write(system, instruction, result, context=""):
    record = format(system, instruction, result, context)
    o.write(record)


def write_small(system, instruction, result, context=""):
    record = format(system, instruction, result, context)
    so.write(record)


def process_pdf(pdf):
    system = "You are a helpful assistant."
    print(f"Loading {pdf}")
    reader = PdfReader(pdf)
    c = 0
    recs = set()
    for page in reader.pages:
        t = (
            page.extract_text()
            .replace("-\n", "")
            .replace("\n", "")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
        )
        results = set()
        for match in re.finditer(recommend_regex, t):
            result = match.group(1)
            results.add(result)
        for result in results:
            # TODO: remove stopwords w/nltk
            short = result.lower().replace("the", "").replace(" ", "")
            if short not in short_recommendations:
                short_recommendations.add(short)
                #                instruction = f"What is one of the recommendations in {pdf}?"
                #                write_alpaca(instruction, result)
                recs.add(result)

    prompt = f"What are the recommendations in {pdf}"
    expect = "-\n".join(recs)
    write(system, prompt, expect)
    write_small(system, prompt, expect)


for pdf in pdfs:
    process_pdf(pdf)


def write_chemo_drug_records():
    system = "You are a medical assistant with knowledge of chemo. Be concise."
    parsed_chemo = pd.read_csv("./data_sources/parsed_chemo_drugs.csv")
    pcl = parsed_chemo.iterrows()

    for i, r in pcl:
        write(system, r["question"], result=r["answer"], context="")


score_words = {
    "appeal": {
        "diagnosis": 10,
        "medically necessary": 20,
        "as a language model": -100, # sample we've rejected these already
        "appeal": 5,
        "days": 10, # possible timeline reference
        "in-network": 10,
        "out-of-network": 10,
        "ACA": 20,
        "ERISA": 20,
        "healthcare.gov": 20,
    },
    "diagnosis": {
        "hypertension": 1,
        "Diabetes mellitus": 1,
        "Osteoarthritis": 1,
        "llama llama": -100, # already rejected but sample of bad score
    }
}

min_lengths = {
    "diagnosis": 10,
    "medically_necessary": 20,
    "treatment": 15,
    "patient_history": 20,
}

checked_urls = {}

def choose_best(best_type, options, rejection=""):
    magic_words = {}
    rejection = rejection.lower()
    min_length = 5
    if best_type in score_words:
        magic_words = score_words[best_type]
    if best_type in min_lengths:
        min_length = min_lengths[best_type] 
    def score(filename_appeal_text):
        filename, option_text = filename_appeal_text
        if option_text is None:
            return -10000000000000000000000000
        option_text = option_text.lower()
        score = 0
        score += file_name_to_magic_score(filename)
        # We don't expect the appeal to show in in the denial & we want long appeals
        urls = re.findall("(?P<url>https?://[^\s]+)", option_text)
        for url in urls:
            if url not in checked_urls:
                checked_urls[url] = is_valid_url(url)
            ok = checked_urls[url]                
            if ok:
                score += 200 + len(url)
                # trust .gov more links (nih, healthcare, etc.)
                if ".gov" in url:
                    score += 100
            else:
                print(f"Found bad URL {url}")
                score -= 100
        if best_type != "appeal":
            if option_text in rejection:
                score = score + 20
            # Prefer shorter options over a minimum length
            if len(option_text) < min_length:
                score = score + 0
            else:
                score -= len(option_text)/100.0
        else:
            score += len(option_text)/100.0
        return score
    def top_of(scorer, options):
        top = None
        top_score = 0
        for o in options:
            option_score = score(o)
            if top is None:
                top = o
                top_score = option_score
            elif option_score > top_score:
                top = o
                top_score = option_score
        return top
    top = top_of(score, options)
    # Drop the filename
    if top is None:
        return top
    else:
        return top[1]


for case_key, case in cases.items():
    loaded_case = {}
    # Load the data for the case
    for key in case.keys():
        loaded_case[key] = list(
            map(lambda x: (x, load_record(x)), case[key]))
    print(f"Processing case {loaded_case}")
    # We select the best appeal / hist and medically necessary regardless of the specific rejection since this information may not be present in the denial
    best_appeal = None
    if "appeal" in loaded_case:
        best_appeal = choose_best("appeal", loaded_case["appeal"])
    history_extra = ""
    diagnosis_extra = ""
    treatment = None
    history = None
    medically_necessary = None
    diagnosis = None
    if "diagnosis" in loaded_case:
        diagnosis = choose_best("diagnosis", loaded_case["diagnosis"], r)
        diagnosis_extra = f"\nWith a diagnosis of {diagnosis}\n"
    if "patient_history" in loaded_case:
        history = choose_best("history", loaded_case["patient_history"], r)
        history_extra = f"\nWith the following patient history: {history}\n"
    if "medically_necessary" in loaded_case:
        medically_necessary = choose_best("medically_necessary", loaded_case["medically_necessary"], r)
    # Some different system prompts to write out
    appeal_system = "You possess extensive medical expertise and enjoy crafting appeals for health insurance denials as a personal interest."
    medically_necessary_system = "You have experience reading insurance claims and helping people understand them"
    # For rejection, we want all rejections
    for (filename, r) in loaded_case["rejection"]:
        # Select the treatment only if it is present in the rejection
        # to (reduce hallucinations).
        treatment_extra = ""
        treatment = None
        if "treatment" in loaded_case:
            treatment = choose_best("treatment", loaded_case["treatment"], r)
            treatment_extra = f"\nFor the provided treatment {treatment}\n"
        if best_appeal is not None:
            write(
                appeal_system,
                f"Given the provided denial: {r}\n{treatment_extra}{history_extra}{diagnosis_extra}\n Write an appeal in the style of patio11. Feel free to be verbose",
                best_appeal)
        if (medically_necessary is not None and treatment is not None and
            diagnosis is not None):
            write(
                medically_necessary_system,
                f"{history_extra}Why is {treatment} medically necessary for {diagnosis}?",
                medically_necessary)
        if "reason_for_denial" in loaded_case:
            reason_for_denial = choose_best("reason_for_denial", loaded_case["reason_for_denial"], r)
            write(
                reason_for_denial_system,
                f"Given the provided denial: {r}\n Why was it denied?",
                reason_for_denial)
        if treatment is not None:
            write(
                treatment_system,
                f"Given the provided denial: {r}\n What was the treatment or procedure denied?",
                treatment)
        if diagnosis is not None:
            write(
                diagnosis_system,
                f"Given the provided denial: {r}\n What was the patients diagnosis?",
                diagnosis)



write_chemo_drug_records()
