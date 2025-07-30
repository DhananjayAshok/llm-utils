from examples.generate_example_data import setup_data
from utils import load_parameters
import click


loaded_parameters = load_parameters()

@click.group()
@click.pass_context
def main(ctx):
    ctx.obj = loaded_parameters
    pass

main.add_command(setup_data, name="setup")

if __name__ == "__main__":
    main()