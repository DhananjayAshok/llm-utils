from vllm import LLM, SamplingParams
import click
import torch


@click.command()
@click.option("--temperature", type=float, default=1)
@click.option("--top_p", type=float, default=1)
@click.pass_obj
def vllm_inference(parameters, temperature, top_p):
    data_df, output_filepath = parameters["output_df"], parameters["output_filepath"]
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=parameters["max_new_tokens"],
                                     stop_strings=parameters["stop_strings"])
    n_gpus = torch.cuda.device_count()
    llm = LLM(model=parameters["model_name"], tensor_parallel_size=n_gpus)
    outputs = llm.generate(data_df[parameters["input_column"]], sampling_params)
    data_df[parameters["output_column"]] = outputs
    data_df[parameters["generation_complete_column"]] = True
    data_df.to_csv(output_filepath, index=False)
    return