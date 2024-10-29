"""
### TODO: Add docstring
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
from peft import PeftModel
from dotenv import load_dotenv
import wandb
from huggingface_hub import login

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
load_dotenv()
wandb.login(key=os.getenv("wandb"))
login(token=os.getenv("huggingface"), add_to_git_credential=True)

def main(model_grade: int, test_set_grade: int) -> None:
    """
    ### TODO: Add docstring
    """
    model_name = "meta-llama/Meta-Llama-3-8B"
    config = AutoConfig.from_pretrained(model_name)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                config=config,
                                                quantization_config=quantization_config,
                                                low_cpu_mem_usage=True,
                                                )
    print("#"*50)
    print("Loaded base model")

    # if model_grade == -1: # ! deprecated with new llama model
    #     current_model_name = f"gpt2-base-eval-on-grade-{test_set_grade}"
    # elif model_grade == 0: # ! deprecated with new llama model
    #     adapters = "williamplacroix/text-simplification/gpt2-2-12-all"
    #     model = PeftModel.from_pretrained(model, adapters)
    #     current_model_name = f"gpt2-2-12-all_eval-on-grade-{test_set_grade}"
    ### TODO refactor with model aliases for longterm maintainability
    if model_grade == 1:
        adapters = "williamplacroix/llama-text-simplification/llama38b-2-12-evens"
        model = PeftModel.from_pretrained(model, adapters)
        current_model_name = f"llama38b-2-12-evens_eval-on-grade-{test_set_grade}"
    else: # * here's where the magic happens
        finetuned_adapter = f"williamplacroix/llama-text-simplification/llama38b-grade-{model_grade}-finetuned"
        model = PeftModel.from_pretrained(model, finetuned_adapter)
        print("Loaded PeFT model")
        print(model)
        print("#"*50)
        current_model_name = f"llama38b-grade-{model_grade}_eval-on-grade-{test_set_grade}"

    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
    wandb.init(project="Graded text simplification evaluation",
               group=f"Grade: {test_set_grade}",
               name=current_model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model.config.pad_token_id = tokenizer.eos_token_id

    tokenized_dataset = load_dataset("williamplacroix/wikilarge-graded", f"grade-{test_set_grade}")

    print(f"Grade {test_set_grade}:", tokenized_dataset)

    data_collator = DataCollatorForSeq2Seq(model=model,
                                           tokenizer=tokenizer,
                                           padding="max_length",
                                           pad_to_multiple_of=8,
                                           max_length=128,
                                           label_pad_token_id=tokenizer.eos_token_id)

    training_args = TrainingArguments(
        logging_strategy="epoch",
        save_strategy="epoch",
        eval_strategy="epoch",
        output_dir="williamplacroix/llama-text-simplification",
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

    print(f"Running evaluation on model_grade: {model_grade}, test_set_grade: {test_set_grade}")
    print("#"*50)

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
    m_grade = int(sys.argv[1])
    t_grade = int(sys.argv[2])
    main(m_grade, t_grade)
