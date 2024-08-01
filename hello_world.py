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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompt = "def hello_world():"

model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
model.to(device)

generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
print(tokenizer.batch_decode(generated_ids)[0])