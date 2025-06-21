# Selection Aware Hyperparameter Tuning for Double Machine Learning in Sample Selection Models

This repository accompanies the Master Thesis in Economics at Ludwig-Maximilians-Universität München (LMU Munich), conducted at the Department of Economics. The thesis investigates hyperparameter tuning in Double Machine Learning (DML) for Sample Selection Models (SSMs), with a focus on improving Average Treatment Effect (ATE) estimation under Missing Not at Random (MNAR) settings.

Special thanks to Prof. Tomasz Olma for his supervision and guidance throughout this work.

---

## Table of Contents

- [Main Contributions](#main-contributions)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
  - [Install Requirements](#1-install-requirements)
  - [Patch DoubleML for SSM](#2-patch-doubleml-for-ssm)
  - [Simulations (Main Study)](#3-simulations-main-study)
  - [Analysis Selection-Aware Tuning](#4-analysis-selection-aware-tuning)
  - [Analysis Study Comparison](#5-analysis-study-comparison)
- [Contact](#contact)

---

## Main Contributions

The main contributions are:
- **Selection-aware tuning procedures:** The thesis introduces new full-sample and on-folds tuning strategies that consistently incorporate estimated selection probabilities into the tuning of treatment and outcome models.  
  *This change is implemented directly in the tuning methods of the DoubleML package's `ssm.py`. Therefore, you must use the `ssm.py` script provided in this repository to ensure correct functionality.*

![spl_method_dml_ssm drawio (1)](https://github.com/user-attachments/assets/3928654c-ea96-4a9c-bf44-67087fb493e0)

- **Comprehensive simulation study:** The thesis provides an extensive simulation-based evaluation of different machine learning methods (Lasso, Random Forest, XGBoost) for nuisance function estimation in DML-SSM. The impact of tuning, model choice, and DGP complexity on ATE estimation is analyzed for MNAR scenario, with results benchmarked against oracle models.

The codebase includes all simulation and analysis tools used in the thesis.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── analysis_selection_aware_tuning/ # Selection-aware vs Default Tuning analysis
│   │   ├── lasso_results/               # Results and analysis for Lasso model
│   │   ├── rf_results/                  # Results analysis for Random Forest model
│   │   ├── xgb_results/                 # Results analysis for XGBoost model
│   ├── analysis_study_comparison/       # Comparative study analysis of Full Sample and On Folds across ML Models
│   │   ├── results_fs/                  # Results and analysis for full-sample tuning
│   │   ├── results_of/                  # Results and analysis for on-folds tuning
│   │   ├── analysis.ipynb               # Main analysis notebook
│   ├── real_life_study/                 # Real-life study analysis
│   │   ├── data/                        # Data files for real-life study
│   │   ├── real_life_study_academic.ipynb # Academic study notebook
│   │   ├── real_life_study_analysis_academic.ipynb # Analysis for academic study
│   │   ├── real_life_study_analysis_vocational.ipynb # Analysis for vocational study
│   │   ├── real_life_study_vocational.ipynb # Vocational study notebook
│   ├── simulations/                     # Main simulation code and utilities
│   │   ├── DGP_functions.py             # Data generating processes
│   │   ├── helpers.py                   # Helper functions
│   │   ├── main.py                      # Main simulation entry point
│   │   ├── oracle_functions.py          # Oracle-related functions
│   ├── ssm.py                           # Custom DoubleML SSM implementation
└── ...                                  # Other files and folders
```

---

## How to Run

### 1. Install Requirements

To run any part of this repository, first install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Patch DoubleML for SSM

   This repository provides a custom `ssm.py` file in `src/ssm.py`.  
   You must manually overwrite the DoubleML package's own `ssm.py` with this file:

   ```bash
   cp src/custom_ssm.py <your_python_env>/lib/pythonX.X/site-packages/doubleml/irm/ssm.py
   ```

   - This step is required for selection-aware tuning to work as described in the thesis.

### 3. Simulations (Main Study)

The simulations directory contains the main code for running experiments and simulations. Follow these steps:

**Run the MPI Simulation**

   Use the following command format:

   ```bash
   mpirun -np <num_processes> python src/simulations/main.py --mar <True/False> --oracle <True/False> --dgp_num <int> --tuning_method <full_sample/split_sample/on_folds> --tune <True/False> --ml_models <model1> <model2> ... --n_sim <num_simulations> --n_obs <num_observations>
   ```

   **Example:**

   ```bash
   mpirun -np 4 python src/simulations/main.py --mar False --oracle False --dgp_num 1 --tuning_method full_sample --tune True --ml_models lasso xgb --n_sim 100 --n_obs 2000
   ```

**Explanation of Parameters**

| Argument           | Description                                                                 | Example                                 |
|--------------------|-----------------------------------------------------------------------------|-----------------------------------------|
| `-np`              | Number of parallel processes (adjust based on resources)                    | `-np 8`                                 |
| `--mar`            | Whether to assume Missing At Random (MAR) (`True`/`False`)                  | `--mar False`                           |
| `--oracle`         | Whether to have Oracle results (`True`/`False`)                             | `--oracle True`                         |
| `--dgp_num`        | DGP number (integer)                                                        | `--dgp_num 1`                           |
| `--tuning_method`  | Type of hyperparameter tuning (`full_sample`/`on_folds`)                    | `--tuning_method on_folds`              |
| `--tune`           | Whether to perform tuning (`True`/`False`)                                  | `--tune True`                           |
| `--ml_models`      | List of ML models to use                                                    | `--ml_models lasso regression xgb rf`   |
| `--n_sim`          | Number of simulations to run for each model                                 | `--n_sim 200`                           |
| `--n_obs`          | Number of observations per simulation                                       | `--n_obs 3000`                          |

   - Logging is automatically handled, and a log file (`simulation.log`) is created to track progress and duration.

### 4. Analysis Selection-Aware Tuning

This directory contains pre-produced CSV files comparing default tuning and selection-aware tuning using both default and custom `ssm.py`. The comparison is performed across three models: Random Forest (RF), Lasso, and XGBoost (XGB).

1. Navigate to `src/analysis_selection_aware_tuning/`.
2. Each machine learning model has its own subdirectory (`lasso_results/`, `rf_results/`, `xgb_results/`) containing precomputed results.
3. Within each subdirectory, you will find a notebook analyzing the results and CSV files that can be used for further study.

### 5. Analysis Study Comparison

This directory evaluates full-sample and on-folds tuning methods using custom `ssm.py` across different machine learning models.

1. Navigate to `src/analysis_study_comparison/`.
2. Open `analysis.ipynb` to explore the comparative study.
3. Results and analysis notebooks for full-sample and on-folds tuning are stored in `results_fs/` and `results_of/` directories, respectively.

---

## Contact

For any questions or further information, please contact:

**Name:** Batuhan Tongarlak  
**Email:** b.tongarlak@campus.lmu.de
