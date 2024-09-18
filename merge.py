import pandas as pd
from datasets import Dataset
from datasets import load_metric
from evaluate import load

from transformers import TrainingArguments, Seq2SeqTrainingArguments
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer, Seq2SeqTrainer
from transformers import GenerationConfig
from transformers import EarlyStoppingCallback
from peft import LoraModel, LoraConfig, PeftModel
import sys
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()
import wandb

wandb.login(key=os.getenv("wandb"))

from huggingface_hub import login
login(token=os.getenv("huggingface"), add_to_git_credential=True)

from evaluate import load
sari = load("sari")


def main():

    data_location = './data/wikilarge/'   
    test_set_grade = int(sys.argv[1])

    os.environ["WANDB_PROJECT"] = f"Graded text simplification evaluation - grade {test_set_grade}"  # name your W&B project
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

    model_name = "openai-community/gpt2"

    train_texts = pd.read_pickle(f'{data_location}train_texts.pkl')
    print("train texts read in")

    grade_groups = train_texts.groupby(['target_grade'])

    datasets = {}
    for i, (grade, group) in enumerate(grade_groups):
        ### [:320] is to limit the number of examples per grade group to 320 for batch subsetting
        # datasets[i] = Dataset.from_pandas(group[['source', 'target', 'target_grade']][:320]).train_test_split(test_size=0.1)
        datasets[i] = Dataset.from_pandas(group[['source', 'target', 'target_grade']]).train_test_split(test_size=0.1)
    print("datasets created")
    

    ### change dataset[N] where N is the grade group you want to train on
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")#, pad_token="eos_token") #pad_token_id=tokenizer.pad_token_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    

    """
    Below function tokenizes parallel corpus into source:target pairs for Seq2Seq training
    """
    # def tokenize_function(examples):
    #     return tokenizer(text=examples["source"], text_target=examples['target'], padding=True, truncation=True, max_length=1024, return_tensors="pt")

    """
    Below function tokenizes parallel corpus into target only inputs for unsupervised fine-tuning
    """
    def tokenize_function(examples):
        return tokenizer(text=examples["target"], text_target=examples["target"], padding=True, truncation=True, max_length=1024, return_tensors="pt")

    config = AutoConfig.from_pretrained(model_name)
    
    base_model = AutoModelForCausalLM.from_pretrained(model_name, config=config)

    adapters = [f"williamplacroix/gpt2-grade-{test_set_grade+1}", f"williamplacroix/gpt2-grade-{test_set_grade-1}"]
    model = PeftModel.from_pretrained(base_model, adapters)
    merged_model = model.merge_and_unload()

    print(merged_model)
    merged_model.config.pad_token_id = tokenizer.eos_token_id

    generation_config = GenerationConfig(max_length=256, 
                                            max_new_tokens=256,
                                            pad_token_id=tokenizer.eos_token_id,)
    generation_config.save_pretrained("./generation_config")
    merged_model.generation_config.pad_token_id = tokenizer.pad_token_id

    tokenized_dataset = datasets[test_set_grade].map(tokenize_function, batched=True, batch_size=32,
                                    remove_columns=['target_grade','source', 'target', '__index_level_0__'])
    print(tokenized_dataset)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="max_length", max_length=128, label_pad_token_id=tokenizer.eos_token_id)

    #train_data_loader = torch.utils.data.DataLoader(tokenized_dataset['train'], batch_size=32, shuffle=True, collate_fn=data_collator)
    # eval_data_loader = torch.utils.data.DataLoader(tokenized_dataset['test'], batch_size=training_args.batch_size, shuffle=False, collate_fn=data_collator)
    # dataloaders = {'train': train_data_loader, 'eval': eval_data_loader}

    # for batch in train_data_loader:
    #     print(batch.keys())
    #     print(batch['input_ids'].shape)
    #     print(batch['attention_mask'].shape)
    #     print(batch['labels'].shape)
    #     print(batch['target_grade'].shape)
    #     break
    print("data collated")

    current_model_name = f"merge-{test_set_grade-1}-with-{test_set_grade+1}_eval-on-grade-{test_set_grade}"

    training_args = Seq2SeqTrainingArguments(
        save_strategy="epoch",
        output_dir=f"./models/{current_model_name}",
        report_to="wandb",  # enable logging to W&B
        run_name=current_model_name,  # name of the W&B run (optional)
        logging_steps=1,  # how often to log to W&B
        #hub_model_id="williamplacroix/text-simplification",  # save the model to the Hub after training
        overwrite_output_dir=True,
        save_safetensors=False, # this is a temporary fix for a bug in the transformers library
        #save_only_model=True,
        save_total_limit=1,
        eval_strategy="epoch",
        learning_rate=1e-5,
        weight_decay=0.01,
        seed=42,
        num_train_epochs=5,
        load_best_model_at_end=True,
        prediction_loss_only=True,
        metric_for_best_model="loss",
        #greater_is_better=False,
        label_names=["labels"],
        include_inputs_for_metrics=True,
        predict_with_generate=True,
        generation_config="./generation_config/generation_config.json",
        generation_max_length=256,
        remove_unused_columns=False,
    )
    
    trainer = Seq2SeqTrainer(
        model=merged_model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=data_collator,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        #compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        
    )

    trainer.evaluate()

    return

if __name__ == "__main__":
    main()