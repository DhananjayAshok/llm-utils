import click
from utils.parameter_handling import load_parameters, compute_secondary_parameters
from inference.huggingface_inference import hf_inference

loaded_parameters = load_parameters()

# Any parameter from your project that you want to be able to change from the command line should be added as an option here
@click.group()
@click.option("--storage_dir", default=loaded_parameters["storage_dir"], help="The directory where the data is stored")
@click.option("--random_seed", default=loaded_parameters["random_seed"], help="The random seed for the project")
@click.option("--log_file", default=loaded_parameters["log_file"], help="The file to log to")
@click.pass_context
def main(ctx, **input_parameters):
    loaded_parameters.update(input_parameters)
    compute_secondary_parameters(loaded_parameters)
    ctx.obj = loaded_parameters


main.add_command(hf_inference)