import json
from os import listdir
from os.path import isfile, join


with open('header.txt') as x: header = x.read()

category = "creative_writing"

raw_dataset = "dataset_src"

data_files = [f for f in listdir(raw_dataset) if (isfile(join(raw_dataset, f)))]

rejection_files = [f for f in data_files if f.endswith("_rejection.txt")]

with open("out/out.jsonl", "w") as o:
    for f in rejection_files:
        with open(join(raw_dataset, f)) as r: rejection = r.read()
        with open(join(raw_dataset, f.replace("rejection", "appeal"))) as a: appeal = a.read()
        prompt = f"{header}{rejection}"
        record = json.dumps({
            "instruction": prompt,
            "context": "",
            "response": appeal,
            "category": category})
        record.replace("\n", "")
        o.write(record)
        o.write("\n")
