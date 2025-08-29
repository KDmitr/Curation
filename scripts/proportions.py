import pandas as pd
import numpy as np
import torch

if torch.cuda.is_available():
    from tabpfn import TabPFNClassifier, TabPFNRegressor
else:
    from tabpfn_client import TabPFNClassifier, TabPFNRegressor

from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.metrics import roc_auc_score, r2_score
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.model_selection import train_test_split

import os
import argparse


def data_exist(train_folder, test_folder, df_name) -> bool:
    train_exist = os.path.exists(f'{train_folder}/{df_name}.csv')
    test_exist  = os.path.exists(f'{test_folder}/{df_name}.csv')

    if not train_exist:
        print(f'No train data for dataset "{df_name}"')
    if not test_exist:
        print(f'No test data for dataset "{df_name}"')

    return train_exist and test_exist


def save_benchmark_csv(output_folder, df_name, benchmark_res, curation):
    if curation == 'yes':
        save_folder = f'{output_folder}/proportions_results_curated/{df_name}/'
    else:
        save_folder = f'{output_folder}/proportions_results_not_curated/{df_name}/'
    os.makedirs(os.path.dirname(save_folder), exist_ok=True)
    benchmark_res.to_csv(save_folder + 'res_df.csv')


def real_synth_proportion(real_data, synth_data, target_name, task_type):
    if len(real_data) < len(synth_data):
        synth_data = synth_data.sample(len(real_data))
    else:
        real_data = real_data.sample(len(synth_data))

    if len(real_data) > 10000: # We need to crop data since TabPFN can't handle more than 10000 samples
        real_data = real_data.sample(10000)
        synth_data = synth_data.sample(10000)
        
    proportions = [70, 50, 30]
    datasets = [None, None, None]

    for i, prop in enumerate(proportions):
        real_proportion = prop
        synth_proportion = 100 - prop

        real_data_prop, _ = train_test_split(real_data,
                                             train_size=real_proportion / 100,
                                             stratify=real_data[target_name] if task_type == 'classification' else None,
                                             shuffle=True,
                                             random_state=42)
        
        synth_data_prop, _ = train_test_split(synth_data,
                                             train_size=synth_proportion / 100,
                                             stratify=synth_data[target_name] if task_type == 'classification' else None,
                                             shuffle=True,
                                             random_state=42)
        
        datasets[i] = pd.concat([real_data_prop, synth_data_prop])
        print(f'Proportion: {real_proportion}-{synth_proportion}, Real: {len(real_data_prop)}, Syn: {len(synth_data_prop)}')

    return datasets[0], datasets[1], datasets[2]


# We leave the classes with the largest number of samples since TabPFN has a limit of no more than 20 classes 
def modify_us_location(train_data, test_data, synth_data): 
    values_to_remove = train_data['bird'].value_counts().sort_values(ascending=True)[:20].keys().to_list()
    train_data = train_data[~train_data['bird'].isin(values_to_remove)]
    test_data = test_data[~test_data['bird'].isin(values_to_remove)]
    synth_data = synth_data[~synth_data['bird'].isin(values_to_remove)]

    return train_data, test_data, synth_data


def train_classification(prop_70_30, prop_50_50, prop_30_70, test_data, target_name):
    proportions = [prop_70_30, prop_50_50, prop_30_70]
    results_mlp = [None, None, None]
    results_xgb = [None, None, None]
    results_tabpfn = [None, None, None]

    X_test = test_data.drop(columns=[target_name])
    y_test = test_data[target_name]

    for i, prop in enumerate(proportions):
        X_train = prop.drop(columns=[target_name])
        y_train = prop[target_name]

        mlp = MLPClassifier(max_iter=10000)
        mlp.fit(X_train, y_train)
        mlp_preds_proba = mlp.predict_proba(X_test)

        if len(y_test.unique()) > 2:
            y_test_bin = label_binarize(y_test, classes=np.sort(y_test.unique()))
            mlp_roc_auc = roc_auc_score(y_test_bin, mlp_preds_proba, multi_class='ovr')
        else:
            if mlp_preds_proba.shape[1] != 1:
                mlp_roc_auc = roc_auc_score(y_test, mlp_preds_proba[:, 1])
            else:
                mlp_roc_auc = roc_auc_score(y_test, mlp_preds_proba)

        xgb = XGBClassifier()
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        xgb.fit(X_train, y_train_encoded)
        xgb_preds_proba = xgb.predict_proba(X_test)
        
        if len(y_test.unique()) > 2:
            xgb_roc_auc = roc_auc_score(y_test_encoded, xgb_preds_proba, multi_class='ovr')
        else:
            if xgb_preds_proba.shape[1] != 1:
                xgb_roc_auc = roc_auc_score(y_test_encoded, xgb_preds_proba[:, 1])
            else:
                xgb_roc_auc = roc_auc_score(y_test_encoded, xgb_preds_proba)

        tabpfn = TabPFNClassifier()
        tabpfn.fit(X_train, y_train)
        tabpfn_preds_proba = tabpfn.predict_proba(X_test)

        if len(y_test.unique()) > 2:
            y_test_bin = label_binarize(y_test, classes=np.sort(y_test.unique()))
            tabpfn_roc_auc = roc_auc_score(y_test_bin, tabpfn_preds_proba, multi_class='ovr')
        else:
            if tabpfn_preds_proba.shape[1] != 1:
                tabpfn_roc_auc = roc_auc_score(y_test, tabpfn_preds_proba[:, 1])
            else:
                tabpfn_roc_auc = roc_auc_score(y_test, tabpfn_preds_proba)

        results_mlp[i] = mlp_roc_auc
        results_xgb[i] = xgb_roc_auc
        results_tabpfn[i] = tabpfn_roc_auc

    return results_mlp, results_xgb, results_tabpfn


def train_regression(prop_70_30, prop_50_50, prop_30_70, test_data, target_name):
    proportions = [prop_70_30, prop_50_50, prop_30_70]
    results_mlp = [None, None, None]
    results_xgb = [None, None, None]
    results_tabpfn = [None, None, None]

    X_test = test_data.drop(columns=[target_name])
    y_test = test_data[target_name]

    for i, prop in enumerate(proportions):
        X_train = prop.drop(columns=[target_name])
        y_train = prop[target_name]

        mlp = MLPRegressor(max_iter=100000)
        mlp.fit(X_train, y_train)
        mlp_preds = mlp.predict(X_test)
        mlp_r2 = r2_score(y_test, mlp_preds)

        xgb = XGBRegressor()
        xgb.fit(X_train, y_train)
        xgb_preds = xgb.predict(X_test)
        xgb_r2 = r2_score(y_test, xgb_preds)

        tabpfn = TabPFNRegressor()
        tabpfn.fit(X_train, y_train)
        tabpfn_preds = tabpfn.predict(X_test)
        tabpfn_r2 = r2_score(y_test, tabpfn_preds)

        results_mlp[i] = mlp_r2
        results_xgb[i] = xgb_r2
        results_tabpfn[i] = tabpfn_r2

    return results_mlp, results_xgb, results_tabpfn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark evaluation")
    parser.add_argument("--data_folder", default=f"./data", help="Data Folder")
    parser.add_argument("--output_folder", default=f"./results", help="Output Folder")
    parser.add_argument("--repeats", default=5, type=int, help="Number of experiments")
    parser.add_argument("--curation", default="no", help='Evaluate results for not curated/curated synthetics (curation - "yes", no curation - "no")')
    
    args = parser.parse_args()

    data_folder = args.data_folder
    output_folder = args.output_folder

    train_folder = f'{data_folder}/train'
    test_folder = f'{data_folder}/test'

    curation = args.curation

    if not os.path.exists(f'{data_folder}/data_info.csv'):
        raise Exception(f'Missing "data_info.csv" in folder "{data_folder}"')
    
    plugins_list = [
                'ctgan',
                'ddpm',
                'gpt4o', 
                'llama',
                'qwen',
                'batch_training' # LLAMA (batch-trained)
                ]
    
    data_info = pd.read_csv(f'{data_folder}/data_info.csv', index_col='df_name').sort_values('row_number')
    df_names = ['breast_cancer', 'diabetes', 'seattle_housing', 'us_location', 'nursery', 'adult'] 

    results = {}

    for df_name in df_names:
        print(f'Starting "{df_name}"\n')
    
        if not data_exist(train_folder, test_folder, df_name):
            print(f'{df_name} skipped (no data)')
            continue

        df_test = pd.read_csv(f'{test_folder}/{df_name}.csv')
        df_train = pd.read_csv(f'{train_folder}/{df_name}.csv')

        target_name = data_info.loc[df_name, 'target_name']
        task_type = data_info.loc[df_name, 'task_type']
        
        for plugin_name in plugins_list:
            print(f'  Processing {plugin_name}')
            if curation == 'yes':
                generated_data_folder = output_folder + f'/generated_data_curated/{df_name}/{plugin_name}/'
            else:
                generated_data_folder = output_folder + f'/generated_data_not_curated/{df_name}/{plugin_name}/'
                
            if len(os.listdir(generated_data_folder)) < args.repeats:
                print(df_name, plugin_name, 
                      f': not enough generated data, need {args.repeats} but got {len(os.listdir(generated_data_folder))}, \
                        skipping experiments...\n')
                continue

            mean_70_30 = []
            mean_50_50 = []
            mean_30_70 = []

            X_syn_combined = None

            for repeat in range(args.repeats):
                X_syn_path = generated_data_folder + f'X_syn_{repeat}.csv'
                X_syn_df = pd.read_csv(X_syn_path)

                X_syn_df = X_syn_df.dropna()

                if plugin_name in ['gpt4o', 'llama','qwen', 'batch_training']:
                    if plugin_name != 'batch_training':
                        X_syn_df = X_syn_df[X_syn_df['real_identifier'] == 0] # for llm generations
                    if X_syn_df.columns[0] == 'Unnamed: 0':
                        X_syn_df = X_syn_df.drop(columns=['Unnamed: 0'])
                    elif X_syn_df.columns[0] == 'Unnamed: 0.1':
                        X_syn_df = X_syn_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])

                    if 'real_identifier' in X_syn_df.columns:
                        X_syn_df = X_syn_df.drop(columns=['real_identifier'])
                
                X_syn_df = X_syn_df[df_train.columns]

                if curation == 'no':
                    if df_name != 'diabetes' and len(X_syn_df) != 100:
                        X_syn_df = X_syn_df.sample(100, random_state=42)
                    elif df_name == 'diabetes' and len(X_syn_df) != 30:
                        X_syn_df = X_syn_df.sample(30, random_state=42)
                
                if X_syn_combined is None:
                    X_syn_combined = X_syn_df
                else:
                    X_syn_combined = pd.concat([X_syn_combined, X_syn_df])

            if df_name == 'nursery' and plugin_name != ['ctgan', 'ddpm', 'batch_training']:
                X_syn_combined['target'] = X_syn_combined['target'].replace({2: 3, 3: 4})

            rare_classes = X_syn_combined[target_name].value_counts()[X_syn_combined[target_name].value_counts() == 1].index.tolist()
            X_syn_combined = X_syn_combined[~X_syn_combined[target_name].isin(rare_classes)] # remove rare classes

            if df_name == 'us_location':
                train_data, test_data, syn_data = modify_us_location(df_train, df_test, X_syn_combined)
                train_70_30, train_50_50, train_30_70 = real_synth_proportion(train_data, syn_data, target_name, task_type)
            else:
                train_70_30, train_50_50, train_30_70 = real_synth_proportion(df_train, X_syn_combined, target_name, task_type)

            if task_type == 'classification':
                if df_name == 'us_location':
                    roc_auc_mlp, roc_auc_xgb, roc_auc_tabpfn = train_classification(train_70_30, train_50_50, train_30_70, test_data, target_name)
                else:
                    roc_auc_mlp, roc_auc_xgb, roc_auc_tabpfn = train_classification(train_70_30, train_50_50, train_30_70, df_test, target_name)
                mean_70_30 = [roc_auc_mlp[0], roc_auc_xgb[0], roc_auc_tabpfn[0]]
                mean_50_50 = [roc_auc_mlp[1], roc_auc_xgb[1], roc_auc_tabpfn[1]]
                mean_30_70 = [roc_auc_mlp[2], roc_auc_xgb[2], roc_auc_tabpfn[2]]
            else:
                r2_mlp, r2_xgb, r2_tabpfn = train_regression(train_70_30, train_50_50, train_30_70, df_test, target_name)
                mean_70_30 = [r2_mlp[0], r2_xgb[0], r2_tabpfn[0]]
                mean_50_50 = [r2_mlp[1], r2_xgb[1], r2_tabpfn[1]]
                mean_30_70 = [r2_mlp[2], r2_xgb[2], r2_tabpfn[2]]

            mean_70_30 = [round(el, 3) for el in mean_70_30]
            mean_50_50 = [round(el, 3) for el in mean_50_50]
            mean_30_70 = [round(el, 3) for el in mean_30_70]

            if df_name not in results:
                results[df_name] = {}

            if plugin_name not in results[df_name]:
                results[df_name][plugin_name] = {
                'mean_70_30': None,
                'mean_50_50': None,
                'mean_30_70': None
            }

            results[df_name][plugin_name]['mean_70_30'] = mean_70_30
            results[df_name][plugin_name]['mean_50_50'] = mean_50_50
            results[df_name][plugin_name]['mean_30_70'] = mean_30_70
        
        model_data = results[df_name]
        scores = pd.DataFrame.from_dict(model_data, orient='index')

        save_benchmark_csv(output_folder, df_name, scores, curation)
        print(f'"{df_name}" benchmark results saved\n')

    print(results)
    
        
        

        




