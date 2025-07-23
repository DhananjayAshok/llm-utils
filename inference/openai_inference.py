from utils.parameter_handling import load_parameters

import base64
import os
import json

import pandas as pd
from openai import OpenAI


class OpenAIInference:
    def __init__(self, variant="gpt-4o-mini", parameters=None):
        raise NotImplementedError
        assert variant in ["gpt-4o", "gpt-4o-mini"]
        self.client = OpenAI()
        self.variant = variant
        if parameters is None:
            parameters = load_parameters()
        self.openai_tmp_dir = parameters["openai_tmp_dir"]
        if not os.path.exists(self.openai_tmp_dir):
            os.makedirs(self.openai_tmp_dir)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    
    def convert_image_text_to_dict_line(self, image_text, max_tokens=10):
        # {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
        image, text = image_text
        image.save(os.path.join(self.openai_tmp_dir, "tmp_image.jpg"))
        image_base64 = self.encode_image(os.path.join(self.openai_tmp_dir, "tmp_image.jpg"))
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}, {"type": "text", "text": text}]}]
        d = {"method": "POST", "url": "/v1/chat/completions", "body": {"model": self.variant, "messages": messages, "max_tokens": max_tokens}}
        return d
    
    def convert_to_dict_line(self, messages, max_tokens=10):
        d = {"method": "POST", "url": "/v1/chat/completions", "body": {"model": self.variant, "messages": messages, "max_tokens": max_tokens}}
    
    def read_batch_results(self, file):
        file_response = self.client.files.content(file)
        # each response will be a json string
        columns = ["idx", "response"]
        data = []
        for line in file_response.text.split("\n"):
            if line:
                d = json.loads(line)
                id = int(d["custom_id"].split("_")[1])
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
        with open(os.path.join(self.openai_tmp_dir, f"id_{batch_name}.txt"), "r") as f:
            batch_id = f.read()
        batch = self.client.batches.retrieve(batch_id)
        if batch.status == "completed":
            batch_file = batch.output_file_id
            return self.read_batch_results(batch_file)
        else:
            print(f"Batch {batch_name} is not completed. Returning None.")
            return None


    def message_call(self, texts, batch_name, ids=None):
        if os.path.exists(os.path.join(self.openai_tmp_dir, f"id_{batch_name}.txt")):
            print(f"Batch {batch_name} already exists. Returning results from file.")
            return self.get_batch_results(batch_name)
        # otherwise
        if texts is None:
            print(f"Got None for image_texts. Returning None.")
            return None
        if ids is not None:
            assert len(texts) == len(ids), "Length of image_texts and ids should be the same."
        requests = []
        for i, image_text in enumerate(image_texts):
            d = self.convert_to_dict_line(image_text)
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
                "description": f"VLM Inference Batch {batch_name}"
            }
        )
        # write the batch id to tmp
        with open(os.path.join(self.openai_tmp_dir, f"id_{batch_name}.txt"), "w") as f:
            f.write(batch.id)
        return None


    def __call__(self, inp_list, batch_name, ids=None, image_texts=False):
        if os.path.exists(os.path.join(self.openai_tmp_dir, f"id_{batch_name}.txt")):
            print(f"Batch {batch_name} already exists. Returning results from file.")
            return self.get_batch_results(batch_name)
        # otherwise
        if inp_list is None:
            print(f"Got None for inp_list. Returning None.")
            return None
        if ids is not None:
            assert len(inp_list) == len(ids), "Length of inp_list and ids should be the same."
        requests = []
        for i, content in enumerate(inp_list):
            if image_texts:
                d = self.convert_image_text_to_dict_line(content)
            else:
                d = self.convert_to_dict_line(content)
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
        # write the batch id to tmp
        with open(os.path.join(self.openai_tmp_dir, f"id_{batch_name}.txt"), "w") as f:
            f.write(batch.id)
        return None
    
    def __str__(self):
        return f"{self.variant}"
    
