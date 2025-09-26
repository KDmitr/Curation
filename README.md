# Synthetic Data Generation and Evaluation Framework

This repository contains a comprehensive framework for generating, curating, and evaluating synthetic data using various machine learning techniques, including generative models and LLM-based approaches.

## 🏗️ Project Structure

```
├── data/                           # Dataset storage and information
│   ├── train/                     # Training datasets
│   ├── test/                      # Test datasets
│   └── data_info.csv             # Metadata about available datasets
├── scripts/                       # Core implementation scripts
│   ├── curation.py               # Multi-objective subset optimization for data curation
│   ├── train_on_real_syn.py     # Training and evaluation on real vs synthetic data
│   ├── proportions.py            # Proportion analysis and evaluation
│   ├── generate_syn/             # Synthetic data generation methods
│   │   ├── fit_and_generate.py   # Main generation pipeline
│   │   ├── generator_syn.py      # Generator implementations
│   │   ├── synthcity_metrics.py  # Evaluation metrics
│   │   ├── vectgan.py           # VectGAN implementation
│   │   └── LLAMA/               # LLAMA-based generation
│   └── generate_syn_cur/         # Curated synthetic data generation
│       ├── llama_fine_tuning.py  # LLAMA model fine-tuning
│       ├── llama_inference_iterative.py # Iterative inference
│       └── prepare_data_prompts.ipynb # Prompt preparation
├── results/                       # Generated results and outputs
│   ├── requirements.txt           # Python dependencies
│   ├── generated_data_curated/   # Curated synthetic datasets
│   ├── generated_data_not_curated/ # Non-curated synthetic datasets
│   ├── proportions_results_curated/ # Curated proportion results
│   ├── proportions_results_not_curated/ # Non-curated proportion results
│   └── real_syn_tested_on_real/  # Real vs synthetic comparison results
└── visuals/                       # Visualization outputs
    ├── regression.png             # Regression task visualizations
    └── classification.png         # Classification task visualizations
```

## 🎯 Project Overview

This framework focuses on synthetic data generation and evaluation across multiple domains:

- **Classification Tasks**: Adult income, breast cancer, credit scoring, diabetes, and more
- **Regression Tasks**: Body fat prediction, CPU performance, housing prices, sea level data
- **Multi-modal Data**: Support for both tabular and text-based datasets

## 🚀 Key Features

### 1. Multi-Objective Data Curation (`curation.py`)
- **Multi-criteria optimization** for optimal training data subsets
- **Pareto front optimization** for ML performance vs privacy trade-offs
- Support for both classification and regression tasks

### 2. Synthetic Data Generation
- **Traditional ML approaches**: CTGAN, Copulas, VectGAN
- **LLM-based generation**: LLAMA fine-tuning and inference
- **Hybrid approaches**: Combining multiple generation methods

### 3. Comprehensive Evaluation Framework
- **ML performance metrics**: ROC-AUC, F1-score, R², MSE
- **Data quality metrics**: Statistical similarity, privacy preservation
- **Comparative analysis**: Real vs synthetic data performance

### 4. Advanced ML Models
- **TabPFN**: Transformer-based few-shot learning
- **XGBoost**: Gradient boosting for structured data
- **Neural Networks**: MLP classifiers and regressors

## 📊 Supported Datasets

The framework includes 25+ datasets covering various domains:

| Dataset | Task Type | Target | Size |
|---------|-----------|---------|------|
| adult | Classification | income | 26,049 |
| breast_cancer | Classification | target | 455 |
| diabetes | Regression | target | 353 |
| iris | Classification | target | 120 |
| wine | Classification | target | 142 |
| seattle_housing | Regression | price | 1,613 |


## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd repo
```

2. **Install dependencies**:
```bash
pip install -r results/requirements.txt
```

## 📋 Dependencies

Key packages include:
- **Core ML**: `scikit-learn`, `torch`, `xgboost`
- **Synthetic Data**: `synthcity`, `ctgan`, `copulas`
- **Evaluation**: `sdmetrics`, `be_great`
- **LLM**: `transformers`, `huggingface-hub`
- **Visualization**: `seaborn`, `matplotlib`

## 🚀 Usage Examples

### 1. Generate Synthetic Data
```python
from scripts.generate_syn.fit_and_generate import generate_synthetic_data

# Generate synthetic data using CTGAN
synth_data = generate_synthetic_data(
    real_data=real_dataset,
    method='ctgan',
    target_column='target'
)
```

### 2. Curate Training Data
```python
from scripts.curation import MultiObjectiveSubsetOptimizer

optimizer = MultiObjectiveSubsetOptimizer(
    task_type='classification',
    use_loo_selection=True,
    pareto_front=True
)

curated_subset = optimizer.optimize(dataset)
```

### 3. Evaluate Real vs Synthetic Performance
```python
from scripts.train_on_real_syn import train_and_evaluate

results = train_and_evaluate(
    train_data=real_data,
    synth_data=synthetic_data,
    test_data=test_data,
    target_name='target'
)
```

## 📈 Results and Outputs

The framework generates comprehensive results in the `results/` directory:

- **Synthetic datasets** with and without curation
- **Performance comparisons** between real and synthetic data
- **Proportion analysis** for data distribution preservation
- **Visualization outputs** for regression and classification tasks

## 🔬 Research Applications

This framework is particularly useful for:
- **Data augmentation** in low-resource scenarios
- **Privacy-preserving ML** through synthetic data
- **Model robustness** evaluation across different data distributions
- **Transfer learning** between real and synthetic domains

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request





