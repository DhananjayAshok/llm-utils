from utils import load_parameters, log_warn, log_error, log_info

import base64
from PIL import Image
from skimage import io
import os
import json
import click

import pandas as pd
from openai import OpenAI


class OpenAIInference:
    def __init__(self, variant="gpt-4o-mini", max_new_tokens=10, parameters=None):
        options = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-5-mini", "gpt-5-nano", "gpt-5", "gpt-4"]
        if variant not in options:
            log_error(f"Variant {variant} not supported. Choose from {options}", parameters)
        self.client = OpenAI()
        self.max_new_tokens = max_new_tokens
        self.variant = variant
        self.parameters = load_parameters(parameters)
        self.openai_tmp_dir = self.parameters["openai_tmp_dir"]
        if not os.path.exists(self.openai_tmp_dir):
            os.makedirs(self.openai_tmp_dir)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    
    def convert_image_text_to_dict_line(self, text, image):
        max_tokens = self.max_new_tokens
        image.save(os.path.join(self.openai_tmp_dir, "tmp_image.jpg"))
        image_base64 = self.encode_image(os.path.join(self.openai_tmp_dir, "tmp_image.jpg"))
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}, {"type": "text", "text": text}]}]
        d = {"method": "POST", "url": "/v1/chat/completions", "body": {"model": self.variant, "messages": messages, "max_tokens": max_tokens}}
        return d
    
    def convert_to_dict_line(self, text):
        max_tokens = self.max_new_tokens
        messages = [{"role": "user", "content": text}]
        d = {"method": "POST", "url": "/v1/chat/completions", "body": {"model": self.variant, "messages": messages, "max_tokens": max_tokens}}
        return d
    
    def read_batch_results(self, file):
        file_response = self.client.files.content(file)
        # each response will be a json string
        columns = ["idx", "response"]
        data = []
        for line in file_response.text.split("\n"):
            if line:
                d = json.loads(line)
                id = d["custom_id"].split("_", 1)[-1]
                response = d['response']['body']["choices"][0]["message"]["content"]
                if "[STOP]" in response:
                    response = response.split("[STOP]")[0].strip()
                data.append([id, response])
        df = pd.DataFrame(data, columns=columns)
        df = df.sort_values(by="idx").reset_index(drop=True)
        return df
    
    def get_batch_status(self, batch_name):
        if not os.path.exists(os.path.join(self.openai_tmp_dir, f"id_{batch_name}.txt")):
            return None
        with open(os.path.join(self.openai_tmp_dir, f"id_{batch_name}.txt"), "r") as f:
            batch_id = f.read()
        batch = self.client.batches.retrieve(batch_id)
        if batch.status == "completed":
            return 1
        else:
            return 0

    def get_batch_results(self, batch_name):
        batch_path = os.path.join(self.openai_tmp_dir, f"id_{batch_name}.txt")
        if not os.path.exists(batch_path):
            return -1
        with open(batch_path, "r") as f:
            batch_id = f.read()
        batch = self.client.batches.retrieve(batch_id)
        if batch.status == "completed":
            batch_file = batch.output_file_id
            return self.read_batch_results(batch_file)
        else:
            log_warn(f"Batch {batch_name} is not completed. Returning None.")
            return 0


    def __call__(self, batch_name, inp_texts=None, inp_images=None, ids=None, use_images=False):
        if os.path.exists(os.path.join(self.openai_tmp_dir, f"id_{batch_name}.txt")):
            log_info(f"Batch {batch_name} already exists. Returning results from file.", self.parameters)
            return self.get_batch_results(batch_name)
        # otherwise
        if inp_texts is None:
            log_warn(f"Got None for inp_texts. Returning None.", self.parameters)
            return None
        if use_images and inp_images is None:
            log_warn(f"Got None for inp_images but use_images is True. Returning None.", self.parameters)
            return None
        if ids is not None:
            assert len(inp_texts) == len(ids), "Length of inp_list and ids should be the same."
        if use_images:
            assert len(inp_texts) == len(inp_images), "Length of inp_texts and inp_images should be the same."
        requests = []
        for i  in range(len(inp_texts)):
            if use_images:
                d = self.convert_image_text_to_dict_line(inp_texts[i], inp_images[i])
            else:
                d = self.convert_to_dict_line(inp_texts[i])
            if ids is not None:
                d["custom_id"] = f"id_{ids[i]}"
            else:
                d["custom_id"] = f"id_{i}"
            requests.append(d)

        with open(os.path.join(self.openai_tmp_dir, f"{batch_name}.json"), "w") as f:
            for i, request_dict in enumerate(requests):
                json_string = json.dumps(request_dict)
                if i != len(requests) - 1:
                    f.write(json_string + '\n')
                else:
                    f.write(json_string)
        log_info(f"Wrote batch file to {os.path.join(self.openai_tmp_dir, f'{batch_name}.json')}", self.parameters)
        batch_input_file = self.client.files.create(
            file=open(os.path.join(self.openai_tmp_dir, f"{batch_name}.json"), "rb"),
            purpose="batch"
        )
         # run the batch
        batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"OpenAI Inference {batch_name}"
            }
        )
        log_info(f"Started batch {batch_name} with id {batch.id}. This may take a while to complete. Check https://platform.openai.com/batches/ for status.", self.parameters)
        # write the batch id to tmp
        with open(os.path.join(self.openai_tmp_dir, f"id_{batch_name}.txt"), "w") as f:
            f.write(batch.id)
        return None
    
    def __str__(self):
        return f"{self.variant}"
    

@click.command()
@click.option("--batch_name", type=str, required=True, help="A name for this batch. Used to store temporary files and track completion.")
@click.pass_obj
def openai_inference(parameters, batch_name):
    max_new_tokens = parameters["max_new_tokens"]
    variant = parameters["model_name"]
    model = OpenAIInference(variant=variant, max_new_tokens=max_new_tokens, parameters=parameters)
    batch_results = model.get_batch_results(batch_name)
    if isinstance(batch_results, int):
        if batch_results == 0:
            return # the batch is still running, should have already been printed to log
        else: # -1 means batch does not exist, so we must create:
            data_df, output_filepath = parameters["output_df"], parameters["output_filepath"]    
            images = None
            if parameters["modality"] == "vlm":
                images = []
                for image_path in data_df[parameters["image_input_column"]].tolist():
                    image = Image.open(image_path) if os.path.isfile(image_path) else Image.fromarray(io.imread(image_path))
                    images.append(image)
            model(batch_name, inp_texts=data_df[parameters["input_column"]].tolist(), inp_images=images, ids=None, use_images=(parameters["modality"] == "vlm"))
    else:
        # batch done and returns dataframe. Just save it
        data_df, output_filepath = parameters["output_df"], parameters["output_filepath"]
        # sort the batch results by idx as int
        batch_results["idx"] = batch_results["idx"].astype(int)
        batch_results = batch_results.sort_values(by="idx").reset_index(drop=True)
        if data_df.shape[0] != batch_results.shape[0]:
            log_warn(f"Critical Warning: Mismatch in number of rows between input data_df ({data_df.shape[0]}) and batch_results ({batch_results.shape[0]}). Will try to align indices", parameters)
            data_df = data_df.reset_index(drop=True)
            for i, row in batch_results.iterrows():
                idx = row['idx']
                if idx not in data_df.index:
                    log_warn(f"Index {idx} from batch_results not found in data_df. Skipping.", parameters)
                    continue
                data_df.at[idx, parameters["output_column"]] = row['response']
        else:
            data_df[parameters["output_column"]] = batch_results["response"].tolist()
        data_df[parameters["generation_complete_column"]] = True
        if data_df[parameters["output_column"]].isnull().any():
            log_warn(f"Some rows in output column {parameters['output_column']} are still null after merging batch results. Dropping rows", parameters)
            orig_size = data_df.shape[0]
            data_df = data_df.dropna(subset=[parameters["output_column"]]).reset_index(drop=True)
            new_size = data_df.shape[0]
            log_info(f"Dropped {orig_size - new_size} rows. New size is {new_size}", parameters)
        data_df.to_json(output_filepath, index=False, orient="records", lines=True)
        log_info(f"Saved output to {output_filepath}", parameters)
        return
