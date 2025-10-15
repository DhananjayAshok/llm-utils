from utils import load_parameters, log_error, log_info
from utils.parameter_handling import compute_secondary_parameters
import click
import os
from inference.huggingface_inference import get_model
from huggingface_hub import HfApi, delete_repo
from huggingface_hub.utils import RepositoryNotFoundError


loaded_parameters = load_parameters()

def get_repo_info(api, repo_id, repo_type):
    try:
        repo_info = api.repo_info(repo_id, repo_type=repo_type)
        return repo_info
    except RepositoryNotFoundError:
        return False


@click.command()
@click.option("--repo_id", type=str, required=True, help="The repository ID on Hugging Face Hub (e.g., username/repo_name)")
@click.option("--model_name", type=str, required=True, help="The path to the model to push to the hub.")
@click.option("--model_kind", type=click.Choice(["gen", "clf"], case_sensitive=False), required=True, help="The kind of model: 'gen' for generative models, 'clf' for classification models.")
@click.option("--quantization", type=click.Choice(["none", "8b", "4b"]), default="none", help="The bitsandbytes quantization method to use.")
@click.option("--modality", type=click.Choice(["lm", "vlm"]), default="lm", help="The modality to use for inference. 'lm' for Language Model, VLM for Vision Language Model.")
@click.option("--overwrite", is_flag=True, default=False, help="If set, will overwrite the existing repository if it exists")
def push_model_to_hub(repo_id, model_name, model_kind, quantization, modality, overwrite):
    """
    Push a model to the Hugging Face Hub.
    """
    compute_secondary_parameters(loaded_parameters)
    parameters = loaded_parameters
    api = HfApi()
    if not os.path.exists(model_name):
        log_error(f"Model path {model_name} does not exist", parameters)
    parameters["model_name"] = model_name
    parameters["modality"] = modality
    parameters["padding_side"] = "left" # This probably doesn't matter
    parameters["dtype"] = "auto" # This probably doesn't matter
    repo_info = get_repo_info(api, repo_id, repo_type="model")
    if repo_info is not False and overwrite:
        log_info(f"Repository {repo_id} already exists. Deleting it as --overwrite is set.", parameters)
        delete_repo(repo_id, repo_type="model")
        repo_info = False
    elif repo_info is not False and not overwrite:
        log_error(f"Repository {repo_id} already exists. Use --overwrite to overwrite it.", parameters)
    model = get_model(parameters, quantization, model_kind)
    processor = parameters["tokenizer"] # I don't know if I need to specifically push the tokenizer and config
    model.push_to_hub(repo_id)
    processor.push_to_hub(repo_id)
    log_info(f"Successfully pushed model at {model_name} to {repo_id}", parameters)


if __name__ == "__main__":
    push_model_to_hub()