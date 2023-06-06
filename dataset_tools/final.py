import json
from os import listdir
from os.path import isfile, join
import re

magic_re = re.compile(r"(.*)(MAGIC[0-9B]*|)(appeal|rejection).txt")

with open('header.txt') as x: header = x.read()

category = "creative_writing"

raw_dataset = "combined-llm-data"

data_files = [f for f in listdir(raw_dataset) if (isfile(join(raw_dataset, f)))]

rejection_files = [f for f in data_files if f.endswith("_rejection.txt")]
appeal_files = [f for f in data_files if f.endswith("_appeal.txt")]

def file_name_to_case(filename):
    groups = magic_re.search(filename)
    return groups.group(1)

def make_appeal_name(rejection, index):
    if index == 0:
        return rejection.replace("rejection", "appeal")
    else:
        return rejection.replace("rejection", f"appeal{i}")

with open("out/out.jsonl", "w") as o:
    for f in rejection_files:
        with open(join(raw_dataset, f), encoding="utf-8") as r: rejection = r.read()
        try:
            with open(join(raw_dataset, make_appeal_name(f, i)), encoding="utf-8") as a: appeal = a.read()
            prompt = f"{header}{rejection}"
            record = json.dumps({
                "instruction": prompt,
                "context": "",
                "response": appeal,
                "category": category})
            record.replace("\n", "")
            o.write(record)
            o.write("\n")
