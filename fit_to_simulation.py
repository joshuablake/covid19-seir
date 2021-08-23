import numpy as np
import os
import pandas as pd
from models import prevalence_model, logPDF_prev
from mcmc import single_block_adapt_mcmc
from time import time

DATA_FILE = os.path.join(os.path.dirname(__file__), "data/simulated/prev_model_simple.csv")

data = pd.read_csv(DATA_FILE)
model = prevalence_model.SingleRegionSingleAgeSEIRCompartmental(100)

known_parameters = {
    "dL": 2,
    "dI": 2,
}
# ordered as: psir, log(i0), pi, beta_rw_sd, betas
true_parameters = [0.15, np.log(1/1000), 0.75, 1] + [0] * model.num_betas
start_values = [0.1, np.log(1/5000), 0.99, 1] + [0] * model.num_betas

log_posterior = logPDF_prev.LogPosteriorOneRegion(
    forward_model=model,
    positive_num=data["num_positive"],
    test_num=data["num_tests"],
    fixed_params = known_parameters
)
print("Estimating: {}".format(log_posterior._estimated_params))

start_time = time()

mcmc = single_block_adapt_mcmc.SingleBlockAMGS(log_posterior, start_values, iterations=2000)
mcmc_out = mcmc.run()
end_time = time()
print("MCMC took {} seconds".format(end_time - start_time))