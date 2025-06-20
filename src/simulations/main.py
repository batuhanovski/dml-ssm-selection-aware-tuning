import argparse
import time
import logging
from mpi4py import MPI
from helpers import *

#simulate_all(True, True, ['lasso', 'regression'], n_sim=10, n_obs=2000).to_csv("results.csv", index=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Run MPI-based simulation.")
    parser.add_argument("--mar", type=str, default=True, help="Enable MAR assumption (True/False)")
    parser.add_argument("--oracle", type=str, default=True, help="Enable Oracle simulations (True/False)")
    parser.add_argument("--dgp_num", type=int, default=True, help="DGP number")    
    parser.add_argument("--tuning_method", type=str, default='full_sample', help="full_sample, split_sample, or on_folds")
    parser.add_argument("--tune", type=str, default='True', help="true or false for tuning")
    parser.add_argument("--ml_models", type=str, nargs='+', default=['lasso', 'regression','xgb', 'rf'], help="List of ML models")
    parser.add_argument("--n_sim", type=int, default=20, help="Number of simulations")
    parser.add_argument("--n_obs", type=int, default=2000, help="Number of observations per simulation")
    return parser.parse_args()

# MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def simulate_more(mar, oracle, tuning_method, dgp_num, ml_models, tune, n_sim, n_obs=2000):
    
    start_time = time.time() 

    local_n_sim = n_sim // size 
    local_results = []

    if rank == 0:

        logging.info(f"Total simulations: {n_sim}, Models: {ml_models}")
        logging.info(f"Total nodes: {size}")
        logging.info(f"Rank 0: Running {local_n_sim} simulations per model.")

    for i in range(rank * local_n_sim, (rank + 1) * local_n_sim):

        result_temp = simulate_one_for_all_models(seed=i, oracle=oracle, mar=mar, tune=tune, tuning_method= tuning_method, ml_models=ml_models, dgp_num=dgp_num, n_obs=n_obs)
        local_results.append(result_temp)
        if rank == 0:
            progress = (i + 1) - (rank * local_n_sim)
            logging.info(f"Rank 0: {progress}/{local_n_sim} completed.")
    
    df_results_local = pd.concat(local_results)

    if rank == 0:
        logging.info("All simulations completed. Gathering results...")

    gathered_results = comm.gather(df_results_local, root=0)

    if rank == 0:
        final_df = pd.concat([df for df in gathered_results if df is not None], axis=0)
        mar_str = "MAR" if mar else "MNAR"
        result_dir = f"results_{mar_str}_dgp{dgp_num}_{tuning_method}.csv"
        final_df.to_csv(result_dir, index=False)
        logging.info(f"Results saved to {result_dir}")

        end_time = time.time()  # Stop timing
        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)  # Get hours
        minutes, seconds = divmod(rem, 60)  # Get minutes & seconds

        logging.info(f"Total execution time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
        logging.info(f"Total simulations: {final_df.shape[0]}")


if __name__ == "__main__":
    args = parse_args()
    args.mar = args.mar == 'True'
    args.oracle = args.oracle == 'True'
    args.tune = args.tune == 'True'
    #LOG
    if rank == 0:
        logging.basicConfig(filename='simulation.log', level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("*********************************************")
        logging.info("Simulation started.")
        logging.info(f"DGPs: {args.dgp_num}")
        logging.info(f"MAR: {args.mar}")
        logging.info(f"ML Models: {args.ml_models}")
        logging.info(f"Oracle: {args.oracle}")
        logging.info(f"Tuning Method: {args.tuning_method}")
        logging.info(f"Tuning: {args.tune}")
        logging.info(f"Number of Simulations: {args.n_sim}")
        logging.info(f"Number of Observations: {args.n_obs}")

    simulate_more(mar = args.mar, oracle = args.oracle, tune=args.tune, tuning_method = args.tuning_method, dgp_num = args.dgp_num, ml_models = args.ml_models, n_sim = args.n_sim, n_obs = args.n_obs)
