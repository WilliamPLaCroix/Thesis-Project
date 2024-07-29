import pandas as pd
from datasets import Dataset
import pickle
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM
from transformers.utils import PaddingStrategy
import torch
import numpy as np
import torch.nn as nn
import random
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from evaluate import load
sari = load("sari")


data_location = './data/wikilarge/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = 'gpt2'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
config = AutoConfig.from_pretrained(model_name,
    max_new_tokens=1024
)

# define new class called training_arguements
class TrainingArguments:
    def __init__(self):
        self.output_dir = "./output/"
        self.evaluation_strategy = "epoch"
        self.batch_size = 32
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-8
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.lr_scheduler_type = "linear"
        self.max_grad_norm = 1.0
        self.max_steps = -1
        self.num_train_epochs = 3
        self.seed = 42
        self.warmup_steps = 0
        self.weight_decay = 0.0
        self.max_sequence_length = 126
        # self.logging_dir = "./logs"
        # self.logging_first_step = False
        # self.logging_steps = 500
        # self.save_steps = 500
        # self.save_total_limit = 1

    def __str__(self):
        print("Training Arguments / Hyperparameters:")
        print("---------------------------------")
        for key, value in self.__dict__.items():
            print(f"| {key}: {value}")
        return "--------------------------------"
training_args = TrainingArguments()
print(training_args)

class FineTuneGPT2(nn.Module):
    def __init__(self, model, tokenizer, training_args):
        super(FineTuneGPT2, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, labels=None):
        # print("input shape:", input_ids.shape)
        # print("attn shape:", attention_mask.shape)
        # print("labels shape:", labels.shape)

        return self.model(input_ids, attention_mask=attention_mask, labels=labels)


def train_test(model, dataloader, optimizer, training):
    """
    Performs a single epoch of training, validation, or testing on the given model using the specified DataLoader.
    This function adapts its behavior based on the 'training' parameter to correctly handle the model's state and
    perform necessary operations such as backpropagation and optimizer updates during training.

    Parameters:
        model (torch.nn.Module): The neural network model to be trained, validated, or tested.
        dataloader (DataLoader): A DataLoader providing batches of data (features and labels) for processing.
        optimizer (torch.optim.Optimizer): The optimizer (AdamW) to use for updating model parameters during training.
        pos_weight (torch.Tensor): A tensor specifying the weight for the positive class to handle class imbalance.
        training (str): A string specifying the mode of operation. Must be 'train', 'validation', or 'test'.

    Returns:
        None if training.
        Cumulative loss (float) if validation.
        A tuple (label_list, prediction_list) containing lists of true labels and predicted labels for
        each sample if testing.
    """
    # BCEWithLogitsLoss combines sigmoid with BCELoss for better stability, and handles class imbalance via pos_weight

    if training == "train":
        model.train()
    elif training == "validation":
        model.eval()
    elif training == "test":
        model.eval()
    else:
        raise ValueError("training argument must be either 'train', 'validation' or 'test'")

    cumulative_loss = 0
    input_list = [] # store inputs accross folds for calculating metrics
    prediction_list = [] # store predictions accross folds for calculating accuracy and f1
    label_list = [] # store labels accross folds for calculating accuracy and f1
    first_loop = True

    for sample in tqdm(dataloader): # iterate over batches in the DataLoader


        sample.to(device)
        input, attention_mask, labels = sample["input_ids"], sample["attention_mask"], sample['labels']
        output = model(input, attention_mask, labels) # forward pass
        loss_value = output.loss
        cumulative_loss += loss_value.item()

        if training == "train":
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
        labels_temp = []
        labels_temp.append(labels.to('cpu').detach().numpy())
        input_list.extend(input.to('cpu').detach().numpy())
        prediction_list.extend(torch.argmax(output.logits, dim=-1).to('cpu').detach().numpy())
        label_list.extend(labels_temp)
        # if first_loop == False:
        #     break
        # first_loop = False


    if training == "train":
        print("cumulative training loss:", cumulative_loss)
        print(compute_metrics((input, torch.argmax(output.logits, dim=-1), labels)))
        return cumulative_loss
    elif training == "validation":
        print("cumulative validation loss:", cumulative_loss)
        print(compute_metrics((input, torch.argmax(output.logits, dim=-1), labels)))
        return cumulative_loss
    elif training == "test":
        return label_list, prediction_list
    else:
        raise ValueError("Ya Done Fuck'd up, son!")

# Training sample
def evaluate(dataloaders, training_args):
    """
    Evaluates neural model's performance on a given task using specified parameters.
    The function preprocesses the data, splits it according to the task, initializes a TuneableModel,
    and trains it. It then evaluates the model on a test set and returns performance metrics.

    The function asserts the task to be one of the predefined tasks and initializes the model based on
    the provided parameters. It supports dynamic pos_weight handling and uses early stopping based on
    validation loss to prevent overfitting.

    Parameters:
        data (pd.DataFrame): The dataset to evaluate the model on.
        parameters (dict): A dictionary containing model hyperparameters and training settings. Expected
            keys include "pos_weight", "batch_size", "alpha", "hidden_size", "dropout", "n_hidden",
            "learning_rate", "beta_1", and "beta_2".
        task (int): An integer indicating the task type. Valid values are 0, 1, 2, and 3, each representing
            a different way of splitting the data for training and testing:
                0 - Known subjects and items with k-fold cross-validation.
                1 - Known subjects and items with leave-one-out cross-validation.
                2 - Held-out subjects, known items.
                3 - Held-out items, known subjects.

    Returns:
        tuple: A tuple containing the accuracy score, F1 score, and confusion matrix of the model evaluated
            on a given test set.
    """

    max_epochs = 1000

    predictions = []
    labels = []
    gpt_new = FineTuneGPT2(model, tokenizer, training_args)
    gpt_new.to(device)
    optimizer = torch.optim.AdamW(gpt_new.parameters(), lr=training_args.learning_rate,
                                  betas=(training_args.adam_beta1, training_args.adam_beta2),
                                  weight_decay=training_args.adam_epsilon)

    train_data_loader = dataloaders['train']
    eval_data_loader = dataloaders['eval']

    max_patience = 2
    last_loss = 1000000
    PATH = f"./models/gpt_new.pt"
    for epoch in range(max_epochs):
        print(f"Epoch {epoch}")
        # training
        train_test(gpt_new, train_data_loader, optimizer, training="train")
        # validation at end of epoch
        with torch.no_grad():
            validation_loss = train_test(gpt_new, eval_data_loader, optimizer, training="validation")

        if validation_loss < last_loss:
            last_loss = validation_loss
            current_patience = 0
        else:
            if current_patience == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': gpt_new.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': last_loss,
                    }, PATH)
            current_patience += 1
        if current_patience == max_patience:
            break

    # # Testing once patience is reached
    # torch.manual_seed(seed)
    # model = TuneableModel(input_size, parameters["hidden_size"], parameters["dropout"], parameters["n_hidden"])
    # gpt_new.to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["learning_rate"], betas=(0.99, 0.99), weight_decay=1e-4)
    # checkpoint = torch.load(PATH)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # with torch.no_grad():
    #     prediction_list, label_list = train_test(model, test_dataloader, optimizer, training="test")
    # predictions.extend(prediction_list)
    # labels.extend(label_list)

    return #compute_metrics() # insert sari ids




def compute_metrics(prediction):
    source_ids, pred_ids, labels_ids = prediction
    sources = []
    labels = []
    predictions = []

    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    source_str = tokenizer.batch_decode(source_ids, skip_special_tokens=True)
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    sources.append(''.join(source_str))
    labels.append([''.join(label_str)])
    predictions.append(''.join(pred_str))


    return sari.compute(sources=sources, predictions=predictions, references=labels)

def tokenize_function(examples):
    return tokenizer(text=examples["source"], text_target=examples['target'], padding=True, max_length=training_args.max_sequence_length, return_tensors="pt")

def find_max_len(tokenized_dataset):
    longest_source = 0
    source = ''
    longest_target = 0
    target = ''
    for dataset in ['train', 'test']:
        for example in tokenized_dataset[dataset]:
            source_len = len(example['input_ids'])
            target_len = len(example['labels'])
            if source_len > longest_source:
                longest_source = source_len
                source = example['input_ids']
            if target_len > longest_target:
                longest_target = target_len
                target = example['labels']
    return max(longest_source, longest_target)


def seed_everything(seed: int):


    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():

    seed_everything(training_args.seed)

    train_texts = pd.read_pickle(f'{data_location}train_texts.pkl')
    print("train texts read in")

    grade_groups = train_texts.groupby(['target_grade'])

    datasets = {}
    for i, (grade, group) in enumerate(grade_groups):
        datasets[i] = Dataset.from_pandas(group[['source', 'target', 'target_grade']]).train_test_split(test_size=0.2)
    print("datasets created")
    
    tokenized_dataset = datasets[6].map(tokenize_function, batched=True, batch_size=training_args.batch_size,
                                      remove_columns=['source', 'target', '__index_level_0__'])

    
    training_args.max_sequence_length = find_max_len(tokenized_dataset)


    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="max_length", max_length=training_args.max_sequence_length, label_pad_token_id=tokenizer.pad_token_id)

    train_data_loader = torch.utils.data.DataLoader(tokenized_dataset['train'], batch_size=training_args.batch_size, shuffle=True, collate_fn=data_collator)
    eval_data_loader = torch.utils.data.DataLoader(tokenized_dataset['test'], batch_size=training_args.batch_size, shuffle=False, collate_fn=data_collator)
    dataloaders = {'train': train_data_loader, 'eval': eval_data_loader}

    for batch in train_data_loader:
        print(batch['input_ids'].shape)
        print(batch['attention_mask'].shape)
        print(batch['labels'].shape)
        print(batch['target_grade'].shape)
        break
    print("data collated")

    evaluate(dataloaders, training_args)
    return

if __name__ == "__main__":
    main()
