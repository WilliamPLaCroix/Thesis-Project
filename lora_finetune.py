"""
:)
"""
import sys
import os
import warnings

from datasets import load_dataset
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
from dotenv import load_dotenv
import wandb
from huggingface_hub import login

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
load_dotenv()
wandb.login(key=os.getenv("wandb"))
login(token=os.getenv("huggingface"), add_to_git_credential=True)

def main(model_grade):
    """
    :)
    """

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
    adapter_name = f"llama38b-grade-{model_grade}-finetuned"
    model_id = "williamplacroix/llama-text-simplification/llama38b-2-12-evens"
    model = PeftModel.from_pretrained(model=model, 
                                      model_id=model_id, 
                                      adapter_name=adapter_name,
                                      is_trainable=True,
                                      )
    
    print("Loaded PeFT model for finetuning")
    current_model_name = f"llama38b-grade-{model_grade}-finetuned"
    wandb.init(project="Graded text simplification training", name=current_model_name)

    print(model)
    model.print_trainable_parameters()
    print("#"*50)

    model.config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    tokenized_dataset = load_dataset("williamplacroix/wikilarge-graded", f"grade-{model_grade}")

    print(f"Grade {model_grade}:", tokenized_dataset)

    data_collator = DataCollatorForSeq2Seq(model=model, tokenizer=tokenizer, padding="max_length", pad_to_multiple_of=8, max_length=128, label_pad_token_id=tokenizer.eos_token_id)
 
    training_args = TrainingArguments(
        logging_strategy="epoch",
        save_strategy="epoch",
        eval_strategy="epoch",
        output_dir="williamplacroix/llama-text-simplification",
        overwrite_output_dir=True,
        report_to="wandb",  # enable logging to W&B
        run_name=current_model_name,  # name of the W&B run (optional)
        logging_steps=1,  # how often to log to W&B
        save_safetensors=False, # this is a kludge fix for a bug in the transformers library
        save_total_limit=1,
        learning_rate=1e-5,
        weight_decay=0.01,
        seed=42,
        num_train_epochs=10, 
        load_best_model_at_end=True,
        remove_unused_columns=False,
    )

    training_args = training_args.set_dataloader(train_batch_size=16, eval_batch_size=16)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.push_to_hub(f"Finished finetuning grade {model_grade}")
    wandb.finish()

if __name__ == "__main__":
    assert int(sys.argv[1]) in {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, "Must include an integer grade as an argument"
    grade = int(sys.argv[1])
    main(grade)
