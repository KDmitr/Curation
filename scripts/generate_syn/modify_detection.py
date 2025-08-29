import os
import pandas as pd

# Path to the metrics folder
root_dir = '/tabular_research/metrics' 

all_combined = []

# Parsing all datasets
for dataset in os.listdir(root_dir):
    dataset_path = os.path.join(root_dir, dataset)
    if not os.path.isdir(dataset_path):
        continue

    model_dfs = []
    model_names = []

    # Parsing all models
    for model in os.listdir(dataset_path):
        model_path = os.path.join(dataset_path, model)
        file_path = os.path.join(model_path, 'res_df.csv')
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        for metric in ['detection.detection_xgb.mean', 'detection.detection_mlp.mean', 'detection.detection_gmm.mean', 'detection.detection_linear.mean']:
            value = df.loc[df['Unnamed: 0'] == metric, 'mean'].values[0]
            modified_value = abs(value - 0.5)
            df.loc[df['Unnamed: 0'] == metric, 'mean'] = modified_value

        output_path = f'/tabular_research/metrics/{dataset}/{model}/res_df.csv'

        df.to_csv(output_path, index=False)