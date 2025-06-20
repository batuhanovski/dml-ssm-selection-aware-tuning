# Selection - Aware Hyperparameter Tuning for Double Machine Learning in Sample Selection Models

This repository accompanies the Master Thesis in Economics at Ludwig-Maximilians-Universität München (LMU Munich), conducted at the Department of Economics. The thesis investigates hyperparameter tuning in Double Machine Learning (DML) for Sample Selection Models (SSMs), with a focus on improving Average Treatment Effect (ATE) estimation under Missing Not at Random (MNAR) settings.

Special thanks to Prof. Tomasz Olma for his supervision and guidance throughout this work.

---

## Table of Contents

- [Main Contributions](#main-contributions)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Logging](#logging)
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
│   ├── analysis/             # Analysis notebooks and scripts
│   │   ├── analysis.ipynb    # Main analysis notebook
│   │   ├── results_fs/       # Results for full-sample tuning
│   │   ├── results_of/       # Results for on-folds tuning
│   ├── simulations/          # Main simulation code and utilities
│   │   ├── DGP_functions.py  # Data generating processes
│   │   ├── helpers.py        # Helper functions
│   │   ├── main.py           # Main simulation entry point
│   │   ├── oracle_functions.py # Oracle-related functions
│   ├── ssm.py                # Custom DoubleML SSM implementation
├── img/                      # Images and diagrams
└── ...                       # Other files and folders
```

---

## How to Run

1. **Install requirements**

   ```bash
   pip install -r requirements.txt
   ```

2. **Patch DoubleML for SSM**

   This repository provides a custom `ssm.py` file in `src/ssm.py`.  
   You must manually overwrite the DoubleML package's own `ssm.py` with this file:

   ```bash
   cp src/ssm.py <your_python_env>/lib/pythonX.X/site-packages/doubleml/irm/ssm.py
   ```
   
   - This step is required for selection-aware tuning to work as described in the thesis.

3. **Run the MPI Simulation**

   Use the following command format:

   ```bash
   mpirun -np <num_processes> python main.py --mar <True/False> --oracle <True/False> --dgp_num <int> --tuning_method <full_sample/split_sample/on_folds> --tune <True/False> --ml_models <model1> <model2> ... --n_sim <num_simulations> --n_obs <num_observations>
   ```

   **Example:**

   ```bash
   mpirun -np 4 python main.py --mar True --oracle False --dgp_num 1 --tuning_method full_sample --tune True --ml_models lasso regression xgb rf --n_sim 100 --n_obs 2000
   ```

### Explanation of Parameters

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

---

## Logging

It creates a log file (`simulation.log`) that shows the progress of each task, total duration, etc.

---

## Contact

For any questions or further information, please contact:

**Name:** Batuhan Tongarlak  
**Email:** b.tongarlak@campus.lmu.de
