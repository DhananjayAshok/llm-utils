from examples.generate_example_data import setup_data, pubmed_process, manymodal_process
from utils import load_parameters
import click


loaded_parameters = load_parameters()

@click.group()
@click.pass_context
def main(ctx):
    ctx.obj = loaded_parameters
    pass

main.add_command(setup_data, name="setup")
main.add_command(pubmed_process, name="pubmed_process")
main.add_command(manymodal_process, name="manymodal_process")

if __name__ == "__main__":
    main()