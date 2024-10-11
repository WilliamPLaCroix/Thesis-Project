from datasets import Dataset, load_dataset

from transformers import TrainingArguments
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer
from transformers import BitsAndBytesConfig

from peft import LoraConfig
from peft import get_peft_model
from peft import PeftModel

import sys
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

import wandb
wandb.login(key=os.getenv("wandb"))

from huggingface_hub import login
login(token=os.getenv("huggingface"), add_to_git_credential=True)


def main(test_set_grade, model_a_proportion):
    
    print(f"Running merge evaluation on test_set_grade: {test_set_grade}")
    
    model_a_proportion = round(model_a_proportion/10, 1)
    model_b_proportion = round(1 - model_a_proportion, 1)
    weights = [model_a_proportion, model_b_proportion]
    
    model_name = "openai-community/gpt2"
    config = AutoConfig.from_pretrained(model_name)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                config=config,
                                                quantization_config=quantization_config,
                                                low_cpu_mem_usage=True,
                                                )
    print("#"*50)
    print("Loaded base model")
    baseline_adapter = "williamplacroix/text-simplification/gpt2-2-12-evens"
    model = PeftModel.from_pretrained(model, baseline_adapter)
    print("Loaded PeFT model")
    model.merge_and_unload()
    print("Merged PeFT model with base")
    
    current_model_name = f"g{test_set_grade-1}-{int(model_a_proportion*100)}_merge_g{test_set_grade+1}-{int(model_b_proportion*100)}_eval-on-g{test_set_grade}"
    print(f"Model name: {current_model_name}")
    print(f"Model proportions: {weights[0]}:{weights[1]}")
    print("#"*50)

    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
    wandb.init(project=f"Graded text simplification evaluation", group=f"Grade: {test_set_grade}", name=current_model_name)

    
    model = PeftModel.from_pretrained(model, f"williamplacroix/text-simplification/gpt2-grade-{test_set_grade-1}-4module", adapter_name="-1")
    print("Loaded trainable PeFT adapter -1")
    model.load_adapter(f"williamplacroix/text-simplification/gpt2-grade-{test_set_grade+1}-4module", adapter_name="+1")
    print("Loaded secondary PeFT adapter +1")

    adapters = ["+1", "-1"]
    
    adapter_name = f"{test_set_grade-1}({int(model_a_proportion*100)})+{test_set_grade+1}({int(model_b_proportion*100)})={test_set_grade}"
    #density = 0.2
    model.add_weighted_adapter(adapters, weights, adapter_name, combination="linear")
    model.set_adapter(adapter_name)

    print("Merged weighted adapters")
    print(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")#, pad_token="eos_token") #pad_token_id=tokenizer.pad_token_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id

    tokenized_dataset = load_dataset("williamplacroix/wikilarge-graded", f"grade-{test_set_grade}")
    
    print(f"Grade {test_set_grade}:", tokenized_dataset)
    
    data_collator = DataCollatorForSeq2Seq(model=model, tokenizer=tokenizer, padding="max_length", pad_to_multiple_of=8, max_length=128, label_pad_token_id=tokenizer.eos_token_id)

    print("data collated")
 
    training_args = TrainingArguments(
        logging_strategy="epoch",
        save_strategy="epoch",
        eval_strategy="epoch",
        output_dir=f"williamplacroix/text-simplification",
        report_to="wandb",  # enable logging to W&B
        run_name=current_model_name,  # name of the W&B run (optional)
        logging_steps=1,  # how often to log to W&B
        save_safetensors=False, # this is a kludge fix for a bug in the transformers library
        learning_rate=1e-5,
        weight_decay=0.01,
        seed=42,
        num_train_epochs=5, 
        load_best_model_at_end=True,
        remove_unused_columns=False,
    )
    
    training_args = training_args.set_dataloader(train_batch_size=32, eval_batch_size=32)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Begin evaluation :)")
    trainer.evaluate()
    wandb.finish()
    return

if __name__ == "__main__":
    test_set_grade = int(sys.argv[1])
    model_a_proportion = round(int(sys.argv[2])/10, 1)
    main(test_set_grade, model_a_proportion)
