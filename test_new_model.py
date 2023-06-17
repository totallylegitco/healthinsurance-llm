from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline)
from dolly.training.generate import generate_response
import torch
from os import listdir
from os.path import isfile, join
from peft import PeftModel, PeftConfig


def do_work(input_model: str,
            fine_tuned: str):

        tokenizer = AutoTokenizer.from_pretrained(fine_tuned, padding_side="left")

        model = AutoModelForCausalLM.from_pretrained(
                fine_tuned, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

        model = PeftModel.from_pretrained(model, input_model)


        raw_dataset = "appeals-llm-data"

        data_files = [f for f in listdir(raw_dataset) if (isfile(join(raw_dataset, f)))]

        rejection_files = [f for f in data_files if f.endswith("_rejection.txt")]

        for f in rejection_files:
                print(f)
                with open(join(raw_dataset, f), encoding="utf-8") as r: print(generate_response("I received a denial: " + r.read() + ". Write an appeal to this denial", model=model, tokenizer=tokenizer))

@click.command()
@click.option("--input-model", type=str, help="Input model to fine tune", required=True)
@click.option("--fine-tuned", type=str, help="Fine tuned model", required=True)
def main(**kwargs):
    print(f"Running w/ {kwargs}")
    do_work(**kwargs)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("main failed")
        raise
