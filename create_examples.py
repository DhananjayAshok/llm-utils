from examples.generate_example_data import setup_data
from utils import load_parameters
import click


loaded_parameters = load_parameters()

@click.group()
@click.pass_context
def main(ctx):
    pass

main.add_command(setup_data, name="generate")

if __name__ == "__main__":
    main()