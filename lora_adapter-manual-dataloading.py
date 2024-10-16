import os
import warnings

import sys

import pandas as pd
from datasets import Dataset, load_dataset

from transformers import TrainingArguments
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer
from transformers import GenerationConfig
from peft import LoraConfig
from peft import get_peft_model
# from peft import prepare_model_for_int8_training
from huggingface_hub import login
import torch
import wandb

from dotenv import load_dotenv
load_dotenv()


wandb.login(key=os.getenv("wandb"))
login(token=os.getenv("huggingface"), add_to_git_credential=True)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_PROJECT"] = "Graded text simplification HF model tampering"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

def main(N):

    #data_location = './data/wikilarge/'

    model_name = "openai-community/gpt2"

    # train_texts = pd.read_pickle(f'{data_location}train_texts.pkl')
    # print("train texts read in")
    # train_texts = train_texts[train_texts['target_grade'] != 0]
    # train_texts = train_texts[train_texts['target_grade'] != 1]
    # print("dropped rows for grades 0 and 1")

    # grade_groups = train_texts.groupby(['target_grade'])

    # datasets = {}
    # for grade, group in grade_groups:
    #     datasets[grade[0]] = Dataset.from_pandas(group[['source', 'target', 'target_grade']]).train_test_split(test_size=0.1, seed=42)
    # print("datasets created")
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")#, pad_token="eos_token") #pad_token_id=tokenizer.pad_token_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # """
    # Below function tokenizes parallel corpus into target only inputs for fine-tuning
    # """
    # def tokenize_function(examples):
    #     return tokenizer(text=examples["target"], text_target=examples["target"], padding=True, truncation=True, max_length=1024, return_tensors="pt")


    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                config=config,
                                                load_in_8bit=True,
                                                torch_dtype=torch.float16)
    print(model)
    model.config.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
                            r=8,
                            lora_alpha=32,
                            target_modules=['lm_head', 'c_attn', 'c_fc', 'c_proj'],
                            task_type="CAUSAL_LM",
                            lora_dropout=0.01,
                            )

    current_model_name = f"gpt2-grade-{N}"

    model = get_peft_model(model=model, peft_config=lora_config)#, adapter_name=current_model_name)
    model.print_trainable_parameters()
    model.config.pad_token_id = tokenizer.eos_token_id

    generation_config = GenerationConfig(max_length=256, 
                                            max_new_tokens=256,
                                            pad_token_id=tokenizer.eos_token_id,)
    generation_config.save_pretrained("./generation_config")
    model.generation_config.pad_token_id = tokenizer.pad_token_id


    # """
    # Detour to push datasets to hub for faster access without repeating preprocessing
    # """
    # for grade, dataset in datasets.items():
    #     preprocessed_dataset = dataset.map(tokenize_function, batched=True, batch_size=32,
    #                                 remove_columns=['target_grade','target', 'source', '__index_level_0__'])
    #     print(preprocessed_dataset)
    #     preprocessed_dataset.push_to_hub(f"williamplacroix/wikilarge-graded", f"grade-{grade}")

    # print("all datasets pushed to hub")
    tokenized_dataset = load_dataset("williamplacroix/wikilarge-graded", f"grade-{N}")
    #tokenized_dataset = datasets[N].map(tokenize_function, batched=True, batch_size=32,
                                    #remove_columns=['target_grade','target', 'source', '__index_level_0__'])

    print(f"Grade {N}:", tokenized_dataset)

    
    data_collator = DataCollatorForSeq2Seq(model=model, tokenizer=tokenizer, padding="max_length", pad_to_multiple_of=8, max_length=128, label_pad_token_id=tokenizer.eos_token_id)

    print("data collated")

    #current_model_name = f"gpt2-grade-{N}"
    current_model_name = "model-for-directory-testing"
 
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
        #fp16=True,
        learning_rate=1e-5,
        weight_decay=0.01,
        seed=42,
        num_train_epochs=1, ### CHANGE THIS BACK TO 5
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

    trainer.train()
    trainer.push_to_hub()
    return

if __name__ == "__main__":
    assert int(sys.argv[1]) in {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, "Must include an integer grade as an argument"
    N = int(sys.argv[1])
    main(N)
