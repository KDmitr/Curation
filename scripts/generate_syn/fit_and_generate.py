import argparse
import os
import joblib

# stdlib
import random
import warnings
warnings.filterwarnings("ignore")

# synthcity absolute
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

import pandas as pd
from vectgan import VECT_GAN_plugin

def data_exist(train_folder, test_folder) -> bool:
    train_exist = os.path.exists(f'{train_folder}/{origin_name}.csv')
    test_exist  = os.path.exists(f'{test_folder}/{origin_name}.csv')

    if not train_exist:
        print(f'No train data for dataset "{origin_name}"')
    
    if not test_exist:
        print(f'No test data for dataset "{origin_name}"')

    return train_exist and test_exist

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Baseline Runner")
    parser.add_argument("--data_folder", default=f"./data", help="Data Folder")
    parser.add_argument("--output_folder", default=f"./results", help="Data Folder")
    parser.add_argument("--optuna_params", default="./optuna", help="Optuna-optimized hyperparameters")
    parser.add_argument("--repeats", default=5, type=int, help="Number of experiments")
    parser.add_argument("--no_rows_filter", action='store_true', help="Include dfs with rows >= 50k")
    
    args = parser.parse_args()

    data_folder = args.data_folder
    output_folder = args.output_folder

    train_folder = f'{data_folder}/train'
    test_folder = f'{data_folder}/test'

    if not os.path.exists(f'{data_folder}/data_info.csv'):
        raise Exception(f'Missing Targets df in folder "{data_folder}"')

    plugins_list = [
                'ctgan',
                'ddpm',
                'goggle',
                'adsgan',
                'dpgan',
                'rtvae',
                'tvae',
                'bayesian_network',
                'vectgan'
                ]
    
    init_kwargs = {
        'goggle' : {},
        'ctgan' : {},
        'ddpm' : {},
        'adsgan' : {},
        'dpgan' : {},
        'rtvae' : {},
        'tvae' : {},
        'bayesian_network' : {},
        'vectgan': {}
    }

    fit_kwargs = {
        'goggle': {},
        'ctgan' : {},
        'ddpm' : {},
        'adsgan' : {},
        'dpgan' : {},
        'rtvae' : {},
        'tvae' : {},
        'bayesian_network' : {},
        'vectgan': {}
    }

    generators = Plugins()

    if 'vectgan' in plugins_list:
        generators.add('vectgan', VECT_GAN_plugin)
        
    print('Plugins Added')

    # Sort dfs according to row number
    data_info = pd.read_csv(f'{data_folder}/data_info.csv', index_col='df_name').sort_values('row_number')

    if not args.no_rows_filter:
        data_info = data_info[data_info['row_number'] <= 50_000]

    df_names = data_info.index

    for df_name in df_names:
        print(f'Start "{df_name}"')

        if df_name.startswith('[reg]') or df_name.startswith('[clf]'):
            origin_name = df_name[6:]
        else:
            origin_name = df_name

        if not data_exist(train_folder, test_folder):
            print(f'"{origin_name}": No train or test, skipped')
            continue
        
        df_train = pd.read_csv(f'{train_folder}/{origin_name}.csv')
        df_test = pd.read_csv(f'{test_folder}/{origin_name}.csv')

        target_name = data_info.loc[df_name, 'target_name']
        task_type   = data_info.loc[df_name, 'task_type']
            
        for plugin_name in plugins_list:
            train_loader = GenericDataLoader(df_train, target_column=target_name)
            test_loader  = GenericDataLoader(df_test, target_column=target_name)
            
            if df_name == 'seattle_housing': # credit_g_params
                params_study = joblib.load(f'results/studies/credit_g/{plugin_name}/credit_g_{plugin_name}_study.pkl')
            
            if df_name == '562_cpu_small': # page_blocks_params
                params_study = joblib.load(f'results/studies/page_blocks/{plugin_name}/page_blocks_{plugin_name}_study.pkl')
            
            if df_name == 'heloc_dataset_v1': # pendigits_params
                params_study = joblib.load(f'results/studies/pendigits/{plugin_name}/pendigits_{plugin_name}_study.pkl')
            
            if df_name in ['online_shoppers_intention', 'XXXX-1', 'XXXX-3']: # nursery_params
                params_study = joblib.load(f'results/studies/nursery/{plugin_name}/nursery_{plugin_name}_study.pkl')
            
            if df_name == 'adult': # magic_04_params
                params_study = joblib.load(f'results/studies/magic04/{plugin_name}/magic04_{plugin_name}_study.pkl')

            if df_name == 'XXXX-2': # diabetes_params
                params_study = joblib.load(f'results/studies/diabetes/{plugin_name}/diabetes_{plugin_name}_study.pkl')

            try:
                init_kwargs[plugin_name] = params_study.best_params
                print(f'Params: {init_kwargs[plugin_name]}')

            except (ValueError):
                print(f"Trials corrupted for {plugin_name}. Skipping...")
                continue
            
            if task_type == 'classification':
                init_kwargs['ddpm']['is_classification'] = True

            print(f'Start training plugin "{plugin_name}"')

            gen = generators.get(plugin_name,
                                compress_dataset=False,
                                strict=False,
                                **init_kwargs[plugin_name])
                
            gen.fit(train_loader, **fit_kwargs[plugin_name])
            print(plugin_name, 'fitted OK')

            # Start of experiments
            generated_data_folder = args.output_folder + f'/generated_data/{df_name}/{plugin_name}/'
            os.makedirs(os.path.dirname(generated_data_folder), exist_ok=True)

            for repeat in range(args.repeats):
                X_syn = gen.generate(len(df_test))

                repeat_save_path = generated_data_folder + f'/X_syn_{repeat}.csv'
                X_syn.dataframe().to_csv(repeat_save_path, index=False)

                print(plugin_name, repeat, 'generated and saved OK')

        print(df_name, ', Done\n')

