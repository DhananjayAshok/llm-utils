from examples.generate_example_data import generate_example_data
import click

@click.group()
def main():
    pass

main.add_command(generate_example_data, name="generate")

if __name__ == "__main__":
    main()