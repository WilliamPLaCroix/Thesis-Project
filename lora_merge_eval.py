import pandas as pd
from datasets import Dataset

from transformers import TrainingArguments
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer
from transformers import GenerationConfig
from peft import LoraConfig
from peft import get_peft_model
from peft import PeftModel
# from peft import prepare_model_for_int8_training

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
    

    model_name = "openai-community/gpt2"

    current_model_name = f"g{test_set_grade-1}-{int(model_a_proportion*100)}_merge_g{test_set_grade+1}-{int(model_b_proportion*100)}_eval-on-g{test_set_grade}"
    print(f"Model name: {current_model_name}")
    print("#"*50)

    os.environ["WANDB_PROJECT"] = "Graded text simplification evaluation"  # name your W&B project
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
    wandb.init(project=f"Graded text simplification evaluation", group=f"Grade: {test_set_grade}", name=current_model_name)

    data_location = './data/wikilarge/'

    

    train_texts = pd.read_pickle(f'{data_location}train_texts.pkl')
    print("train texts read in")
    train_texts = train_texts[train_texts['target_grade'] != 0]
    train_texts = train_texts[train_texts['target_grade'] != 1]
    print("dropped rows for grades 0 and 1")

    grade_groups = train_texts.groupby(['target_grade'])

    datasets = {}
    for grade, group in grade_groups:
        datasets[grade[0]] = Dataset.from_pandas(group[['source', 'target', 'target_grade']]).train_test_split(test_size=0.1, seed=42)
    print("datasets created")
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")#, pad_token="eos_token") #pad_token_id=tokenizer.pad_token_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    """
    Below function tokenizes parallel corpus into target only inputs for fine-tuning
    """
    def tokenize_function(examples):
        return tokenizer(text=examples["target"], text_target=examples["target"], padding=True, truncation=True, max_length=1024, return_tensors="pt")

    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                config=config)
    
    model = PeftModel.from_pretrained(model, f"williamplacroix/gpt2-grade-{test_set_grade-1}", adapter_name="-1")
    _ = model.load_adapter(f"williamplacroix/gpt2-grade-{test_set_grade+1}", adapter_name="+1")

    

    adapters = ["+1", "-1"]
    weights = [model_a_proportion, model_b_proportion]
    adapter_name = f"{test_set_grade-1}({int(model_a_proportion*100)})+{test_set_grade+1}({int(model_b_proportion*100)})={test_set_grade}"
    #density = 0.2
    model.add_weighted_adapter(adapters, weights, adapter_name)
    model.set_adapter(adapter_name)


    #model = model.merge_and_unload()
    print(model)

    model.config.pad_token_id = tokenizer.eos_token_id

    tokenized_dataset = datasets[test_set_grade].map(tokenize_function, batched=True, batch_size=32,
                                    remove_columns=['target_grade','target', 'source', '__index_level_0__'])

    print(tokenized_dataset)
    
    data_collator = DataCollatorForSeq2Seq(model=model, tokenizer=tokenizer, padding="max_length", pad_to_multiple_of=8, max_length=128, label_pad_token_id=tokenizer.eos_token_id)

    print("data collated")
 
    training_args = TrainingArguments(
        logging_strategy="epoch",
        save_strategy="epoch",
        eval_strategy="epoch",
        output_dir=f"./models/{current_model_name}",
        report_to="wandb",  # enable logging to W&B
        run_name=current_model_name,  # name of the W&B run (optional)
        logging_steps=1,  # how often to log to W&B
        #hub_model_id="williamplacroix/text-simplification",  # save the model to the Hub after training
        overwrite_output_dir=True,
        save_safetensors=False, # this is a kludge fix for a bug in the transformers library
        #save_only_model=True,
        save_total_limit=1,
        fp16=True,
        learning_rate=1e-5,
        weight_decay=0.01,
        seed=42,
        num_train_epochs=5,
        load_best_model_at_end=True,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.evaluate()
    wandb.finish()
    return

if __name__ == "__main__":
    test_set_grade = int(sys.argv[1])
    model_a_proportion = round(int(sys.argv[2])/10, 1)
    main(test_set_grade, model_a_proportion)
