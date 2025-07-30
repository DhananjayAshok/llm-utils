from examples.generate_example_data import generate_example_data
from utils import load_parameters
import click


loaded_parameters = load_parameters()

@click.group()
@click.pass_context
def main(ctx):
    pass

main.add_command(generate_example_data, name="generate")

if __name__ == "__main__":
    main()