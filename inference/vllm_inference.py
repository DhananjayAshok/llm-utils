from vllm import LLM, SamplingParams
import click
from inference.huggingface_inference import get_data
import torch


@click.command()
@click.option("--model", type=str, required=True, help="The model to use for generation")
@click.option('--data_path', type=str, required=True)
@click.option('--output_path', type=str, required=True)
@click.option("--data_path", type=str, required=True)
@click.option("--output_csv_path", type=str, required=True)
@click.option("--input_column", type=str, default="input")
@click.option("--output_column", type=str, default="output")
@click.option("--stop_strings", type=str, default="[STOP]")
@click.option("--max_new_tokens", type=int, default=10)
@click.option("--temperature", type=float, default=1)
@click.option("--top_p", type=float, default=1)
@click.pass_obj
def vllm_generate(parameters, model, data_path, output_path, input_column, output_column, stop_strings, max_new_tokens, temperature, top_p):
    data = get_data(data_path, input_column, output_column, parameters)
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, stop_strings=stop_strings)
    n_gpus = torch.cuda.device_count()
    llm = LLM(model=model, tensor_parallel_size=n_gpus)
    outputs = llm.generate(data[input_column], sampling_params)
    data[output_column] = outputs
    data["inference_completed"] = True
    data.to_csv(output_path, index=False)
    return