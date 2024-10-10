import pandas as pd
from datasets import Dataset, load_dataset

from transformers import TrainingArguments
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig

from peft import LoraConfig
from peft import get_peft_model
from peft import PeftModel, PeftConfig

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


def main(model_grade):

    os.environ["WANDB_PROJECT"] = "Graded text simplification training"  # name your W&B project
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

    model_name = "openai-community/gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")#, pad_token="eos_token") #pad_token_id=tokenizer.pad_token_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(model_name)

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                config=config,
                                                quantization_config=quantization_config,
                                                low_cpu_mem_usage=True,
                                                )
    print("#"*50)
    print("Loaded base model:")
    #print(model)
    model.config.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
                            r=8,
                            lora_alpha=32,
                            target_modules=['lm_head', 'c_attn', 'c_fc', 'c_proj'],
                            task_type="CAUSAL_LM",
                            lora_dropout=0.01,
                            )
    adapter_config = PeftConfig.from_pretrained("~/Thesis-Project/adapter_config.json")
    model = PeftModel.from_pretrained(model=model, 
                                      #model_id="williamplacroix/text-simplification/gpt2-2-12-evens",
                                      config=adapter_config,
                                      adapter_name="williamplacroix/text-simplification/gpt2-2-12-evens",
                                      is_trainable=False,
                                      )

    #print("#"*50)
    print("Loaded PeFT model:")
    #print(model)
    model.merge_and_unload()
    #print("#"*50)
    print("Merged PeFT model with base:")
    #print(model)
    model.print_trainable_parameters()
    current_model_name = f"gpt2-grade-{model_grade}-4module"

    wandb.init(project=f"Graded text simplification training", name=current_model_name)

    model = get_peft_model(model=model, peft_config=lora_config, adapter_name=current_model_name)
    #print("#"*50)
    print("Loaded trainable PeFT model:")
    print(model)
    model.print_trainable_parameters()
    print("#"*50)
    model.config.pad_token_id = tokenizer.eos_token_id

    generation_config = GenerationConfig(max_length=256, 
                                            max_new_tokens=256,
                                            pad_token_id=tokenizer.eos_token_id,)
    generation_config.save_pretrained("./generation_config")
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    tokenized_dataset = load_dataset("williamplacroix/wikilarge-graded", f"grade-{model_grade}")

    print(f"Grade {model_grade}:", tokenized_dataset)

    data_collator = DataCollatorForSeq2Seq(model=model, tokenizer=tokenizer, padding="max_length", pad_to_multiple_of=8, max_length=128, label_pad_token_id=tokenizer.eos_token_id)
 
    training_args = TrainingArguments(
        logging_strategy="epoch",
        save_strategy="epoch",
        eval_strategy="epoch",
        output_dir=f"williamplacroix/text-simplification",
        overwrite_output_dir=True,
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

    trainer.train()
    trainer.push_to_hub(f"Finished training grade {model_grade}")
    wandb.finish()
    return

if __name__ == "__main__":
    assert int(sys.argv[1]) in {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, "Must include an integer grade as an argument"
    model_grade = int(sys.argv[1])
    main(model_grade)
