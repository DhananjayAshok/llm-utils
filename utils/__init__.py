from utils.parameter_handling import load_parameters
from utils.log_handling import log_error, log_info, log_warn, log_dict
from utils.hash_handling import write_meta, add_meta_details
from utils.fundamental import file_makedir

import wandb


def get_history(run_name):
    wandb.login()
    api = wandb.Api()
    run = api.run("huggingface/"+run_name)
    history = run.history()
    return history