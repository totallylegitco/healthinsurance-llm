import json
from os import listdir
from os.path import isfile, join


with open('header.txt') as x: header = x.read()

category = "creative_writing"

data_files = [f for f in listdir("dataset_src") if (isfile(join(mypath, f)))]

rejection_files = [f for f in data_files if f.endsWith("_rejection.txt")]

with open("out/data.json", "w") as o:
    for f in rejection_files:
        with open(join(mypath, f)) as r: rejection = r.read()
        with open(join(mypath, f.replace("rejection", "appeal"))) as a: appeal = a.read()
        prompt = f"{header}{rejection}"
        record = json.dumps({
            "instruction": prompt,
            "context": "",
            "response": appeal,
            "category": category})
        o.write(record)
