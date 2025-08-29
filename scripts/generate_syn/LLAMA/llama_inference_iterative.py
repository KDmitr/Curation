import torch
from unsloth import FastLanguageModel
import os
import gc
from tqdm import tqdm
import pandas as pd
import logging
import random
import re 

def get_latest_checkpoint_path(df_name, default_models_path):
    all_checkpoints = os.listdir(default_models_path)
        
    df_checkpoints = [checkpoint for checkpoint in all_checkpoints if df_name in checkpoint]
    df_dates_checkpoints = [checkpoint.removeprefix(f'{df_name}_') for checkpoint in df_checkpoints]
    latest_date = sorted(df_dates_checkpoints)[-1]

    latest_checkpoint_steps = os.listdir(f"{default_models_path}/{df_name}_{latest_date}")
    checkpoints_steps = [int(chkpt.split('-')[-1]) for chkpt in latest_checkpoint_steps]
    latest_steps = sorted(checkpoints_steps)[-1]
    
    latest_checkpoint_path = f'{default_models_path}/{df_name}_{latest_date}/checkpoint-{latest_steps}'
    return latest_checkpoint_path

def check_text_pandas_format(generated_text, 
                            batched,
                            gt_N_cols,
                            gt_cols_names, 
                            separator=' is ',
                            ):
    if batched:
        generated_text = [t for batch in generated_text for t in batch.split('\n') ]
        generated_text = [re.sub("^Sample \d+: ", "", t) for t in generated_text if t]

    sorted_gt_cols_names = sorted(gt_cols_names)
    gt_lower_cols_names  = sorted([str.lower(gt_col) for gt_col in gt_cols_names]
)
    approved_text = []
    for row in generated_text:
        enc = row.split(', ') # [col_1 is val_1, ..., col_n is val_n]
        if len(enc) != gt_N_cols:
            continue

        cols, vals = [], []
        for e in enc:
            if separator not in e: #not enough 'is'
                break 

            col_val_pair = e.split(separator)
            if len(col_val_pair) != 2: #too much 'is'
                break

            col, val = col_val_pair

            try:
                float(val)
            except:
                break  #not float value

            cols.append(col.lstrip().rstrip())
            vals.append(val.replace(',', ''))
        
        if len(cols) != gt_N_cols or len(vals) != gt_N_cols:  #incorrect number of cols
            continue
        
        sorted_cols = sorted(cols)
        lower_cols  = sorted([str.lower(col) for col in sorted_cols])

        if sorted_cols == sorted_gt_cols_names:
            approved_text.append(row)

        elif lower_cols == gt_lower_cols_names:
            syn_col_2_gt_col_dict = {}
            for syn in sorted_cols:
                for gt in sorted_gt_cols_names:
                    if str.lower(gt) == str.lower(syn):
                        syn_col_2_gt_col_dict[syn] = gt

            updated_row = [f'{syn_col_2_gt_col_dict[col]} is {val}' for col, val in zip(cols, vals)]
            updated_row = ', '.join(updated_row)
            approved_text.append(updated_row)

        else:   #wrong cols names    
            continue      
                
    return approved_text

logging.basicConfig(
    filename='inference.log',
    encoding='utf-8',
    level=logging.INFO, 
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )

max_seq_length = 3000 
dtype = None 
load_in_4bit = True 


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

sorted_dict_path = ''
# PAFT orders for datasets
with open(sorted_dict_path, 'r') as f:
    sorted_order_dict = eval(f.read())

data_info_path = ''
data_info = pd.read_csv(data_info_path, index_col='df_name')

PAFT_ORDER = True
STRATEGIES = [
    'standard',
    'batch', 
    'batch_anon', 
    ]


for strategy in STRATEGIES:   
    for df_name in data_info.index:
        DEFAULT_MODELS_PATH   = f"outputs/{strategy}/{'PAFT' if PAFT_ORDER else 'GREAT'}"

        latest_checkpoint_path = get_latest_checkpoint_path(df_name, DEFAULT_MODELS_PATH)

        if 'anon' in strategy:
            N_cols = data_info.loc[df_name, 'col_number']
            cols = [f'col{i}' for i in range(1, N_cols + 1)]
            order = ','.join(cols)
        else:
            order = sorted_order_dict[df_name]

        if 'batch' in strategy:
            N_samples_at_a_time = 10
        else:
            N_samples_at_a_time = 1

        N_experiments = 5
        data_all_rows = data_info.loc[df_name, 'row_number'] + data_info.loc[df_name, 'df_test_len']
        
        Ns_to_generate = [100]
        if data_all_rows >= 500:
            Ns_to_generate.append(500)

        # if 'batch' in STRATEGY:
        #     default_batch_size = 10
        # else:
        #     if df_name == 'breast_cancer':
        #         default_batch_size = 8
        #     else:
        #         default_batch_size = 64
        if df_name == 'breast_cancer':
            default_batch_size = 8
        else:
            default_batch_size = 64

        generated_per_iter = default_batch_size * N_samples_at_a_time

        for N_gen in Ns_to_generate:
            DEFAULT_GEN_DATA_PATH = f"./generated_texts_iter/{N_gen}/{strategy}/{'PAFT' if PAFT_ORDER else 'GREAT'}"

            generated_text_path = f'{DEFAULT_GEN_DATA_PATH}/{df_name}'
            if not os.path.exists(generated_text_path):
                os.makedirs(generated_text_path)

            for experiment in range(N_experiments):    
                if os.path.exists(f'{generated_text_path}/X_syn_{experiment}.txt'):
                    print(df_name, f'X_syn_{experiment} already exists')
                    logging.info(f'{df_name} X_syn_{experiment} already exists')
                    continue

                model, tokenizer = FastLanguageModel.from_pretrained(
                        latest_checkpoint_path,
                        max_seq_length = max_seq_length,
                        dtype = dtype,
                        load_in_4bit = load_in_4bit,
                    )

                FastLanguageModel.for_inference(model)
                
                curr_generated = 0

                print(df_name, 'Start generation number', experiment)
                logging.info(f'{df_name} Start generation number {experiment}')
                
                ITERATION_LIMIT = 5
                no_progress_iteration = 0
                with torch.no_grad():
                    generated_text = []
                    while curr_generated < N_gen:
                        batch_size = default_batch_size

                        prompt = alpaca_prompt.format(f"Generate {N_samples_at_a_time} synthetic samples giving an order of the columns. There must be no columns that were not provided in order!", # instruction
                                                        order,
                                                        "", # output - leave this blank for generation!
                                                    )
                        prompts = [prompt] * batch_size

                        inputs = tokenizer(
                            prompts,
                            return_tensors = "pt").to("cuda")

                        if df_name == 'breast_cancer':
                            max_new_tokens = 256 * 3 * 4
                        else:
                            max_new_tokens = 256 * 3

                        outputs = model.generate(input_ids = inputs.input_ids, attention_mask = inputs.attention_mask,
                                                max_new_tokens=max_new_tokens, use_cache = True)

                        model_output_text = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                       
                        approved_text  = check_text_pandas_format(model_output_text,
                                                                  batched='batch' in strategy,
                                                                  gt_N_cols=data_info.loc[df_name, 'col_number'],
                                                                  gt_cols_names=sorted(order.split(',')),
                                                                  )
                        
                        if curr_generated + len(approved_text) > N_gen:
                            approved_text = random.sample(approved_text, N_gen - curr_generated)

                        if len(approved_text) == 0:
                            no_progress_iteration += 1
                        else:
                            no_progress_iteration = 0

                        curr_generated += len(approved_text)

                        print(df_name, strategy, experiment, ': approved for', len(approved_text))
                        logging.info(f'{df_name} {strategy} {experiment} : approved for {len(approved_text)}')

                        generated_text += approved_text

                        inputs, outputs = None, None
                        gc.collect()
                        torch.cuda.empty_cache()

                        if no_progress_iteration == ITERATION_LIMIT:
                            print('No progress iteration limit exceeded')
                            break

                    if len(generated_text) != N_gen:
                        print(df_name, f'Generation {experiment} FAILED')
                        logging.info(f'{df_name} Generation {experiment} FAILED')

                    else:
                        with open(f'{generated_text_path}/X_syn_{experiment}.txt', 'w+') as f:
                            for line in generated_text:
                                f.write(f"{line}\n")

                        print(df_name, f'Generation {experiment} OK')
                        print(df_name, f'Synthetic {experiment} saved\n')

                        logging.info(f'{df_name} Generation {experiment} OK')
                        logging.info(f'{df_name} Synthetic {experiment} saved')
                        
                        
                model, tokenizer = None, None
                gc.collect()
                torch.cuda.empty_cache()