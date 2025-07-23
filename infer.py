import click
from utils.parameter_handling import load_parameters, compute_secondary_parameters
from utils import log_error
from inference.huggingface_inference import hf_inference
from inference.vllm_inference import vllm_inference
from inference.inference_utils import handle_files

loaded_parameters = load_parameters()

@click.group()
@click.option("--model_name", required=True)
@click.option("--dtype", type=click.Choice(["auto", "float32", "float16", "bfloat16"]), default="auto", help="The data type for the model")
@click.option("--input_file", required=True, help="The path to the input file. Must be either a CSV, json lines or parquet file.")
@click.option("--output_file", default=None, help="The path to the output file. If not provided, it will be set to the input file path with '_output' appended before the file extension.")
@click.option("--input_column", default="input", help="The column in the input file to use as input for the model")
@click.option("--generation_complete_column", default="inference_completed", help="The column in the output file to indicate if the inference is completed")
@click.option("--output_column", default="output", help="The column in the output file to store the model's output")
@click.option("--max_new_tokens", type=int, default=10, help="The maximum number of new tokens to generate")
@click.option("--stop_strings", default=["[STOP]"], multiple=True, help="Strings that will stop the generation when encountered")
@click.option("--ignore_checkpoint", is_flag=True, help="If set, will ignore any existing checkpoint and start from scratch")
@click.option("--random_seed", default=loaded_parameters["random_seed"], help="The random seed for the project")
@click.option("--log_file", default=loaded_parameters["log_file"], help="The file to log to")
@click.pass_context
def main(ctx, **input_parameters):
    loaded_parameters.update(input_parameters)
    compute_secondary_parameters(loaded_parameters)
    output_df, output_filepath = handle_files(input_file=input_parameters["input_file"],
                                              output_file=input_parameters["output_file"],
                                              input_column=input_parameters["input_column"],
                                              generation_complete_column=input_parameters["generation_complete_column"],
                                              output_column=input_parameters["output_column"],
                                              ignore_checkpoint=input_parameters["ignore_checkpoint"],
                                              parameters=loaded_parameters)
    for key in ["output_df", "output_filepath"]:
        if key in loaded_parameters:
            log_error(f"{key} is already present in the loaded parameters", loaded_parameters)
    loaded_parameters["output_df"] = output_df
    loaded_parameters["output_filepath"] = output_filepath
    ctx.obj = loaded_parameters


main.add_command(hf_inference, name="hf")
main.add_command(vllm_inference, name="vllm")

if __name__ == "__main__":
    main()