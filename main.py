from utils.parameter_handling import load_parameters, compute_secondary_parameters
import click


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

# Implement custom commands as functions in a separate file in the following way:
"""
@click.command()
@click.option("--example_option")
@click.pass_obj
def example_command(parameters, example_option):
    # have access to parameters here with any additional arguments that are specific to the script
    pass
"""
# Then add the custom command to the main group like this:
"""
main.add_command(example_command, name="example_command")
"""

if __name__ == "__main__":
    main()