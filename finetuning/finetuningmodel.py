import transformers
import textwrap
from transformers import LlamaTokenizer, LlamaForCausalLM
import os
import sys
from typing import List
import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
import transformers
#import fire
import torch
from datasets import load_dataset
import pandas as pd
import json 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pylab import rcParams
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:516"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = torch.device('mps')

free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
max_memory = f"{free_in_GB-1}GB"
n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}

print("Free In GB: ", free_in_GB)
print("Max Memory: ", max_memory)
print("N GPUs: ", n_gpus)
print("Max Memory: ", max_memory)

sns.set(rc={'figure.figsize':(10, 7)})
sns.set(rc={'figure.dpi':100})
sns.set(style='white', palette='muted', font_scale=1.2)
 
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

'''
df = pd.read_csv("merged_data_withFinalAspects_1.csv")
#print(df.head())
#pos_aspects = df['VALIDATED POSITIVE ASPECTS'].fillna('').to_dict(orient='record')
#neg_aspects = df['VALIDATED NEGATIVE ASPECTS'].fillna('').to_dict(orient='record')

#df['FINAL_ASPECTS'] = df['VALIDATED POSITIVE ASPECTS'].fillna('')

#aspects = dict(zip(df['pos_aspects'], df['neg_aspects']))
#print(pos_aspects)
#print(neg_aspects)

df['FINAL_ASPECTS'] = df.apply(lambda row: f"{row['VALIDATED POSITIVE ASPECTS']}{', ' + row['VALIDATED NEGATIVE ASPECTS'] if pd.notna(row['VALIDATED NEGATIVE ASPECTS']) and row['VALIDATED NEGATIVE ASPECTS'] != '' else ''}", axis=1)
df1 = df[['COMMENTS', 'FINAL_ASPECTS']].copy()
df1['INSTRUCTION'] = "Identify aspects from the input data or review provided"
df1.fillna('',inplace=True)
df1 = df1.rename(columns={'COMMENTS': 'input','FINAL_ASPECTS': 'output', 'INSTRUCTION': 'instruction'})
data = df1.to_dict(orient='records')
#print(df1.to_dict(orient='records'))

with open("trainingData.json", "w") as f:
   json.dump(data, f)
'''
BASE_MODEL = "baffo32/decapoda-research-llama-7B-hf"

model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map = "auto",
    #device_map="mps",#device_map = "auto"
    offload_folder="offload",
    offload_state_dict = True
    #quantization_config=bnb_config  //Remove the comment here and above if you want to use specific quantization
)
'''
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    max_memory=max_memory,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
    torch_dtype=torch.float16,
)
'''
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token_id = (
    0
)
tokenizer.padding_side = "left"
data = load_dataset("json", data_files="trainingData.json")
#data["train"]

def generate_prompt(data_point):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501 ignore comment
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""

CUTOFF_LEN = 64 #1024

def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt

train_val = data["train"].train_test_split(
    test_size=200, shuffle=True, seed=42
)
train_data = (
    train_val["train"].map(generate_and_tokenize_prompt)
)
val_data = (
    train_val["test"].map(generate_and_tokenize_prompt)
)
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

BATCH_SIZE = 4   #128
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 300
OUTPUT_DIR = "test"

model = prepare_model_for_int8_training(model)


config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()


training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=10,
    max_steps=TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    #fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    output_dir=OUTPUT_DIR,
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="tensorboard"
)

data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)
torch.cuda.empty_cache()
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_arguments,
    data_collator=data_collator
)
model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict()
    )
).__get__(model, type(model))

#model = torch.compile(model)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
