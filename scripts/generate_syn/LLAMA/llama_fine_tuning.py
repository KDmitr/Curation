import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from unsloth import FastLanguageModel
import torch
from datasets import load_from_disk
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import pandas as pd
from datetime import date

PAFT_ORDER = True

data_info_path = ''
data_info = pd.read_csv(data_info_path, index_col='df_name').sort_values('row_number')

STRATEGIES = ['standard', 'batch', 'batch_anon']

for df_name in data_info.index:
    for strategy in STRATEGIES:
        max_seq_length = 3000 
        dtype = None 
        load_in_4bit = True 

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "unsloth/Llama-3.2-3B",
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )

        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""

        EOS_TOKEN = tokenizer.eos_token 
        def formatting_prompts_func(examples):
            instructions = examples["instruction"]
            inputs       = examples["input"]
            outputs      = examples["output"]
            texts = []
            for instruction, input, output in zip(instructions, inputs, outputs):
                text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
                texts.append(text)
            return { "text" : texts, }
        

        today_dt = str(date.today())

        data_train_path = f"./data/datasets_for_{strategy}_FT/{'PAFT' if PAFT_ORDER else 'GREAT'}/{df_name}_prompts_train"
        output_dir = f"./outputs/{strategy}/{'PAFT' if PAFT_ORDER else 'GREAT'}/{df_name}_{today_dt}"
        
        dataset = load_from_disk(data_train_path)
        dataset = dataset.map(formatting_prompts_func, batched = True,)

        model = FastLanguageModel.get_peft_model(
            model,
            r = 16, 
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, 
            bias = "none",    
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
            use_rslora = False, 
            loftq_config = None,
        )

        num_train_epochs = 30

        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset,
            dataset_text_field = "text",
            max_seq_length = max_seq_length,
            dataset_num_proc = 2,
            args = TrainingArguments(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
           
                num_train_epochs = num_train_epochs,
                warmup_ratio = 0.10,

                learning_rate = 2e-4,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = 0.1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                save_strategy='steps',
                save_steps=0.25,

                output_dir = output_dir,
                report_to = "none",
            ),
        )

        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        trainer_stats = trainer.train()

        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory         /max_memory*100, 3)
        lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

        print(f'------------ {df_name}, {strategy} OK ------------')