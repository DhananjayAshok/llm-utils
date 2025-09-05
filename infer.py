import click
from utils.parameter_handling import load_parameters, compute_secondary_parameters
from utils import log_error, log_warn
from inference.huggingface_inference import hf_inference
from inference.batch_size_determination import infer_batch_size
from inference.vllm_inference import vllm_inference
from inference.inference_utils import handle_files

loaded_parameters = load_parameters()

@click.group()
@click.option("--model_name", required=True, help="The name of the model to use for inference. This can be a Hugging Face model name or a local path to a model.")
@click.option("--modality", type=click.Choice(["lm", "vlm"]), default="lm", help="The modality to use for inference. 'lm' for Language Model, VLM for Vision Language Model.")
@click.option("--input_file", required=True, help="The path to the input file. Must be either a CSV, json lines or parquet file.")
@click.option("--max_new_tokens", type=int, default=None, help="The maximum number of new tokens to generate")
@click.option("--dtype", type=click.Choice(["auto", "float32", "float16", "bfloat16"]), default="auto", help="The data type for the model")
@click.option("--output_file", default=None, help="The path to the output file. If not provided, it will be set to the input file path with '_output' appended before the extension. Output file is always a JSON lines file")
@click.option("--input_column", default="input", help="The column in the input file to use as text input for the model")
@click.option("--image_input_column", default="image", help="The column in the input file to use as image input for the model. Only used for VLMs.")
@click.option("--generation_complete_column", default="inference_completed", help="The column in the output file to indicate if the inference is completed")
@click.option("--output_column", default="output", help="The column in the output file to store the model's output")
@click.option("--num_return_sequences", type=int, default=1, help="The number of sequences to return for each input. Set to 1 for single output.")
@click.option("--temperature", type=float, default=None, help="The temperature to use for sampling. Higher values lead to more random outputs.")
@click.option("--do_sample", is_flag=True, default=False, help="If set, will use sampling instead of greedy decoding.")
@click.option("--top_p", type=float, default=None, help="The cumulative probability for nucleus sampling. Set to 1 for no nucleus sampling.")
@click.option("--top_k", type=int, default=None, help="The number of highest probability vocabulary tokens to keep for top-k-filtering. Set to -1 for no top-k filtering.")
@click.option("--stop_strings", default=["[STOP]"], multiple=True, help="Strings that will stop the generation when encountered")
@click.option("--ignore_checkpoint", is_flag=True, help="If set, will ignore any existing checkpoint and start from scratch")
@click.option("--random_seed", default=loaded_parameters["random_seed"], help="The random seed for the project")
@click.option("--log_file", default=loaded_parameters["log_file"], help="The file to log to")
@click.pass_context
def main(ctx, **input_parameters):
    input_parameters["stop_strings"] = list(input_parameters["stop_strings"])
    for default_parameter in ["temperature", "top_p", "top_k"]:
        if input_parameters[default_parameter] is None:
            input_parameters.pop(default_parameter)

    if not input_parameters["do_sample"]:
        for sampling_parameter in ["temperature", "top_p", "top_k"]:
            if sampling_parameter in input_parameters:
                log_warn(f"Did not get --do_sample flag, so ignoring {sampling_parameter} parameter", input_parameters)
            input_parameters[sampling_parameter] = None


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
main.add_command(infer_batch_size, name="infer_batch_size")

if __name__ == "__main__":
    main()