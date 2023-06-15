import click
import logging
import torch, einops
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoTokenizer,
    TrainingArguments
)
from peft.tuners.lora import LoraLayer

from trl import SFTTrainer


logger = logging.getLogger(__name__)

device_map = {"": 0}

def create_and_prepare_model(
        input_model: str,
        qlora_4bit: bool,
        qlora_8bit: bool):
    compute_dtype = getattr(torch, "float16")

    bnb_config = None
    peft_config = None
    model = None
    if qlora_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "query_key_value"
            ],
        )
        model = AutoModelForCausalLM.from_pretrained(
            input_model, quantization_config=bnb_config, device_map=device_map, trust_remote_code=True
        )
    elif qlora_8bit:
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            input_model, load_in_8bit=True, device_map=device_map, trust_remote_code=True
        )

        peft_config = LoraConfig(
            lora_alpha=32,
            lora_dropout=0.05,
            r=16,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "query_key_value"
            ],
        )

    tokenizer = AutoTokenizer.from_pretrained(input_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer


def train(local_output_dir: str,
          input_model: str,
          training_dataset: str,
          qlora_4bit: bool,
          qlora_8bit: bool):

    print(f"Loading {training_dataset}")
    dataset = load_dataset(training_dataset, keep_in_memory=True, streaming=False)
    print(dataset)

    print(f"Loading initial model... 4bit {qlora_4bit} 8bit {qlora_8bit}")
    model, peft_config, tokenizer = create_and_prepare_model(
        input_model,
        qlora_4bit,
        qlora_8bit)
    model.config.use_cache = False

    print("Training...")
    training_arguments = TrainingArguments(
        output_dir=local_output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=10,
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True,
        max_grad_norm=0.3,
        max_steps=10000,
        warmup_ratio=0.03,
        # Disable because somehow we have a streaming dataset.
        group_by_length=False,
        lr_scheduler_type="constant",
    )


    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        peft_config=peft_config,
        dataset_text_field="text",
#        max_seq_length=10614784,
        tokenizer=tokenizer,
        args=training_arguments,
        max_seq_length=2048,
    )

    trainer.train()


@click.command()
@click.option("--input-model", type=str, help="Input model to fine tune", required=True)
@click.option("--local-output-dir", type=str, help="Write directly to this local path", default="./results")
@click.option("--training-dataset", type=str, required=True, help="Path to dataset for training", default="./out_oa")
@click.option("--qlora-4bit", type=str, help="Use 4bit mode", default=False)
@click.option("--qlora-8bit", type=str, help="Use 8bit mode", default=False)
def main(**kwargs):
    print(f"Running w/ {kwargs}")
    train(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    try:
        main()
    except Exception:
        logger.exception("main failed")
        raise
