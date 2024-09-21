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
# from peft import prepare_model_for_int8_training

import sys
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_PROJECT"] = "Graded text simplification training"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

import wandb
wandb.login(key=os.getenv("wandb"))

from huggingface_hub import login
login(token=os.getenv("huggingface"), add_to_git_credential=True)


def main():

    data_location = './data/wikilarge/'

    model_name = "openai-community/gpt2"

    N = int(sys.argv[1])

    train_texts = pd.read_pickle(f'{data_location}train_texts.pkl')
    print("train texts read in")

    grade_groups = train_texts.groupby(['target_grade'])

    datasets = {}
    for i, (grade, group) in enumerate(grade_groups):
        datasets[i] = Dataset.from_pandas(group[['source', 'target', 'target_grade']]).train_test_split(test_size=0.1)
    print("datasets created")
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")#, pad_token="eos_token") #pad_token_id=tokenizer.pad_token_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    """
    Below function tokenizes parallel corpus into target only inputs for unsupervised fine-tuning
    """
    def tokenize_function(examples):
        return tokenizer(text=examples["target"], text_target=examples["target"], padding=True, truncation=True, max_length=1024, return_tensors="pt")


    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                config=config,
                                                load_in_8bit=True,
                                                torch_dtype=torch.float16)
    print(model)
    model.config.pad_token_id = tokenizer.eos_token_id

    #model = prepare_model_for_int8_training(model)

    lora_config = LoraConfig(
                            r=8,
                            lora_alpha=32,
                            target_modules=['lm_head'],
                            task_type="CAUSAL_LM",
                            lora_dropout=0.01,
                            )

    model = get_peft_model(model=model, peft_config=lora_config, adapter_name=f"gpt2-grade-{N}")
    model.print_trainable_parameters()
    model.config.pad_token_id = tokenizer.eos_token_id

    generation_config = GenerationConfig(max_length=256, 
                                            max_new_tokens=256,
                                            pad_token_id=tokenizer.eos_token_id,)
    generation_config.save_pretrained("./generation_config")
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    tokenized_dataset = datasets[N].map(tokenize_function, batched=True, batch_size=32,
                                    remove_columns=['target_grade','source', '__index_level_0__'])
    tokenized_dataset['train'].rename_column("target", "label")
    tokenized_dataset['test'].rename_column("target", "label")
    print(tokenized_dataset)
    
    data_collator = DataCollatorForSeq2Seq(model=model, tokenizer=tokenizer, padding="max_length", pad_to_multiple_of=8, max_length=128, label_pad_token_id=tokenizer.eos_token_id)

    train_data_loader = torch.utils.data.DataLoader(tokenized_dataset['train'], batch_size=32, shuffle=True, collate_fn=data_collator)
    
    for batch in train_data_loader:
        print(batch.keys())
        print(batch['input_ids'].shape)
        print(batch['attention_mask'].shape)
        print(batch['labels'].shape)
        print(batch['target_grade'].shape)
        break

    
    print("data collated")

    current_model_name = f"gpt2-grade-{N}"

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
        #prediction_loss_only=True,
        #metric_for_best_model="loss",
        #label_names=["labels"],
        #include_inputs_for_metrics=True,
        #predict_with_generate=True,
        #generation_config="./generation_config/generation_config.json",
        #generation_max_length=256,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        #data_collator=data_collator,
        tokenizer=tokenizer,
        
    )

    trainer.train()
    trainer.push_to_hub(current_model_name)
    return

if __name__ == "__main__":
    main()
