import os
import yaml
from .log_handling import get_logger, log_error


def load_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def compute_secondary_parameters(params):
    params["data_dir"] = os.path.join(params["storage_dir"], "data")
    params["model_dir"] = os.path.join(params["storage_dir"], "models")
    params["log_dir"] = os.path.join(params["storage_dir"], "logs")
    params["tmp_dir"] = os.path.join(params["storage_dir"], "tmp")
    for dir in ["data_dir", "model_dir", "log_dir"]:
        if not os.path.exists(params[dir]):
            os.makedirs(params[dir])
    if "log_file" not in params:
        log_file = os.path.join(params["log_dir"], "log.txt")
        params["log_file"] = log_file
    else:
        # check if log_file is a child of log_dir, but handle silly // vs / cases
        log_dir_str = params["log_dir"].replace("//", "/")
        log_file_str = params["log_file"].replace("//", "/")
        if not log_file_str.startswith(log_dir_str):
            log_file = os.path.join(params["log_dir"], params["log_file"])
            params["log_file"] = log_file
    logger = get_logger(filename=params["log_file"])
    params["logger"] = logger


def load_parameters():
    """
    Loads the parameters for the project from configs/private_vars.yaml and any other yaml files in the configs directory.

        :return: A dictionary of parameters
    """
    essential_keys = ["storage_dir"]
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    params = {"project_root": project_root}
    logger = get_logger() # TODO: Make log parameters. 
    config_files = os.listdir(os.path.join(project_root, "configs"))

    if "private_vars.yaml" not in config_files:
        log_error(logger, "Please create private_vars.yaml in the configs directory")
    for file in config_files:
        if not file.endswith(".yaml") and file != "README.md":
            log_error(logger, f"Please remove {file} from the configs directory. Only yaml files that hold project parameters should be present")        
        configs = load_yaml(os.path.join(project_root, "configs", file))
        for key in configs:
            if key in params:
                log_error(logger, f"{key} is present in multiple config files. At least one of which is {file}. Please remove the duplicate")
        params.update(configs)

    for key in params:
        if params[key] == "PLACEHOLDER":
            log_error(logger, f"{key} is currently the placeholder value in private_vars.yaml. Please set it")
    for essential_key in essential_keys:
        if essential_key not in params:
            log_error(logger, f"Please set {essential_key} in one of the config yamls")
    # check if there are any .py files in storage_dir, if so, log error
    if os.path.exists(params['storage_dir']):
        if any([f.endswith(".py") for f in os.listdir(params["storage_dir"])]):
            log_error(logger, f"There are .py files in the storage_dir {params['storage_dir']}. It is recommended to set a path which has nothing else inside it to avoid issues.")
    else:
        os.makedirs(params['storage_dir'])
        logger.info(f"Created storage directory {params['storage_dir']}")
    compute_secondary_parameters(params)
    return params
