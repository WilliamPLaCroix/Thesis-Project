"""
:)
"""
import os
import warnings

from datasets import Dataset, concatenate_datasets, DatasetDict
from transformers import TrainingArguments
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer
from transformers import BitsAndBytesConfig

# from peft import LoraConfig
# from peft import get_peft_model
from peft import PeftModel#, PeftConfig
import pandas as pd
from dotenv import load_dotenv
import wandb
from huggingface_hub import login

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
load_dotenv()
wandb.login(key=os.getenv("wandb"))
login(token=os.getenv("huggingface"), add_to_git_credential=True)

def main():
    """
    :)
    """

    data_location = './data/wikilarge/'
    model_name = "meta-llama/Meta-Llama-3-8B" # llama38b


    train_texts = pd.read_pickle(f'{data_location}train_texts.pkl')
    print("train texts read in")

    train_texts = train_texts[train_texts['target_grade'] != 0]
    train_texts = train_texts[train_texts['target_grade'] %2 == 0]
    print("dropped rows for odd grades and 0")

    grade_groups = train_texts.groupby(['target_grade'])

    datasets = []
    for grade, group in grade_groups:
        print(f"Creating dataset for grade {grade}")
        datasets.append(Dataset.from_pandas(group[['source', 'target', 'target_grade']]).train_test_split(test_size=0.1, seed=42))
    print("datasets created")

    train_sets = [dataset["train"] for dataset in datasets]
    test_sets = [dataset["test"] for dataset in datasets]
    train_dataset = concatenate_datasets(train_sets)
    train_dataset = train_dataset.shuffle(seed=42)
    test_dataset = concatenate_datasets(test_sets)
    test_dataset = test_dataset.shuffle(seed=42)
    merged_dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})
    print("datasets merged: ", merged_dataset)

    os.environ["WANDB_PROJECT"] = "Graded text simplification training"  # name your W&B project
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

    model_name = "meta-llama/Meta-Llama-3-8B"

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
    print("Loaded base model")
    #print(model)
    model.config.pad_token_id = tokenizer.eos_token_id

    # lora_config = LoraConfig(
    #                         r=8,
    #                         lora_alpha=32,
    #                         target_modules=['lm_head', 'c_attn', 'c_fc', 'c_proj'],
    #                         task_type="CAUSAL_LM",
    #                         lora_dropout=0.01,
    #                         )

    #adapter_name = "llama38b-2-12-evens"
    model_id = "./williamplacroix/text-simplification/checkpoint-6044/llama38b-2-12-evens"
    model = PeftModel.from_pretrained(model=model, 
                                      model_id=model_id, 
                                      #adapter_name=adapter_name,
                                      is_trainable=True,
                                      )
    
    print("Loaded PeFT model for finetuning")
    current_model_name = "llama38b-2-12-evens"

    print(model)
    model.print_trainable_parameters()
    print("#"*50)

    model.config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id


    data_collator = DataCollatorForSeq2Seq(model=model, tokenizer=tokenizer, padding="max_length", pad_to_multiple_of=8, max_length=128, label_pad_token_id=tokenizer.eos_token_id)
 
    training_args = TrainingArguments(
        logging_strategy="epoch",
        save_strategy="epoch",
        eval_strategy="epoch",
        output_dir="williamplacroix/text-simplification",
        report_to="wandb",  # enable logging to W&B
        run_name=current_model_name,  # name of the W&B run (optional)
        logging_steps=1,  # how often to log to W&B
        save_safetensors=False, # this is a kludge fix for a bug in the transformers library
        learning_rate=1e-5,
        weight_decay=0.01,
        seed=42,
        num_train_epochs=1, 
        load_best_model_at_end=True,
        remove_unused_columns=False,
    )

    training_args = training_args.set_dataloader(train_batch_size=16, eval_batch_size=16)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=merged_dataset['train'],
        eval_dataset=merged_dataset['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.save_model()
    trainer.push_to_hub("Finished llama38b 2-12 grades: evens pretraining")

if __name__ == "__main__":
    main()
