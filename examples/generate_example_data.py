from utils import log_error, log_info, log_warn, log_dict
from datasets import load_dataset
import click
import zipfile
import pandas as pd
import os


split_map = {"train": "train", "val": "validation", "test": "test", "dev": "validation", "validation": "validation"}


class SquadExample:
    context_1 = "(Regarding University of Notre Dam) Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary."
    question_1 = "What is the architectural character of the University of Notre Dam?"
    answer_1 = "Catholic character"
    short_question_1 = question_1
    short_answer_1 = answer_1
    long_question_1 = "Write a short sentence on the architectural character of the University of Notre Dam, with special reference on the materials used."
    long_answer_1 = "The University of Notre Dam has a Catholic character, with a gold dome and a golden statue of the Virgin Mary on top of the Main Building."
    context_2 = "(Regarding Kathmandu) The National Museum is located in the western part of Kathmandu, near the Swayambhunath stupa in an historical building."
    question_2 = "Which stupa is near the National Museum in Kathmandu?"
    answer_2 = "Swayambhunath stupa"
    short_question_2 = question_2
    short_answer_2 = answer_2
    long_question_2 = "Describe the location of the National Museum in Kathmandu, complete with references to landmarks around it."
    long_answer_2 = "The National Museum is located in the western part of Kathmandu, near the Swayambhunath stupa in an historical building."

    qa_gen_val_instruction = f"Generate a question and answer pair from the context:"
    qa_gen_val_instruction = qa_gen_val_instruction + "\nContext: " + context_1 + "\nQuestion: " + question_1 + "\nAnswer: " + answer_1 + " [STOP]"
    qa_gen_val_instruction = qa_gen_val_instruction + "\nContext: " + context_2 + "\nQuestion: " + question_2 + "\nAnswer: " + answer_2 + " [STOP]"
    qa_gen_val_instruction = qa_gen_val_instruction + "\nContext: "

    # set up the short and long answer instructions
    qa_gen_short_instruction = f"Generate a short QA pair from the context:"
    qa_gen_short_instruction = qa_gen_short_instruction + "\nContext: " + context_1 + "\nQuestion: " + short_question_1 + "\nAnswer: " + short_answer_1 + " [STOP]"
    qa_gen_short_instruction = qa_gen_short_instruction + "\nContext: " + context_2 + "\nQuestion: " + short_question_2 + "\nAnswer: " + short_answer_2 + " [STOP]"
    qa_gen_short_instruction = qa_gen_short_instruction + "\nContext: "

    qa_gen_long_instruction = f"Generate a long QA pair from the context:"
    qa_gen_long_instruction = qa_gen_long_instruction + "\nContext: " + context_1 + "\nQuestion: " + long_question_1 + "\nAnswer: " + long_answer_1 + " [STOP]"
    qa_gen_long_instruction = qa_gen_long_instruction + "\nContext: " + context_2 + "\nQuestion: " + long_question_2 + "\nAnswer: " + long_answer_2 + " [STOP]"
    qa_gen_long_instruction = qa_gen_long_instruction + "\nContext: "



def setup_squad(parameters):
    """
    Loads the SQuAD dataset and sets up the following files:
        qa_gen_val.csv: contains columns: [context_id, input] which prompts a LM to generate a question, answer pair from the validation set contexts
            We finetune on these QA pairs to see if knowledge can be absorbed by the model.
        qa_gen_short_csv: same columns and contexts as above, but with prompts that ask for short answers
            We use this to test the contrastive learning approaches in this repo.
        qa_gen_long_csv: same columns and contexts as above, but with prompts that ask for long answers
            We use this to test the contrastive learning approaches in this repo.
        val_qa: contains columns: [input, output] where the input is a question and the output is the answer from the validation set.
    """
    log_info("Setting up SQuAD dataset...", parameters)
    df = load_dataset("rajpurkar/squad", split="validation").to_pandas()
    df["title"] = df["title"].apply(lambda x: x.replace("_", " "))
    df["context"] = "(Regarding "+ df["title"] + ") " + df["context"]
    contexts = df["context"].unique()
    columns = ["context_id", "input"]
    qa_gen_val = []
    qa_gen_short = []
    qa_gen_long = []
    for i, context in enumerate(contexts):
        qa_gen_val.append([i, SquadExample.qa_gen_val_instruction + context + "\nQuestion: "])
        qa_gen_short.append([i, SquadExample.qa_gen_short_instruction + context + "\nQuestion: "])
        qa_gen_long.append([i, SquadExample.qa_gen_long_instruction + context + "\nQuestion: "])
    qa_gen_val_df = pd.DataFrame(qa_gen_val, columns=columns)
    qa_gen_short_df = pd.DataFrame(qa_gen_short, columns=columns)
    qa_gen_long_df = pd.DataFrame(qa_gen_long, columns=columns)
    save_dir = parameters["data_dir"]
    os.makedirs(save_dir, exist_ok=True)
    qa_gen_val_df.to_csv(os.path.join(save_dir, "squad_qa_gen_val.csv"), index=False)
    qa_gen_short_df.to_csv(os.path.join(save_dir, "squad_qa_gen_short.csv"), index=False)
    qa_gen_long_df.to_csv(os.path.join(save_dir, "squad_qa_gen_long.csv"), index=False)
    log_info("SQuAD dataset setup complete. Files saved in: " + save_dir)


class ManyModalQAExample:
    colour_question_1 = "What are the primary colours of the Starry Night?"
    colour_answer_1 = "Blue and yellow"
    colour_question_2 = "What is the colour of the hat in the traditional Nepalese Topi?"
    colour_answer_2 = "Red"

    shape_question_1 = "What is the shape of the hat in the traditional Nepalese Topi?"
    shape_answer_1 = "Cone shape"
    shape_question_2 = "What is the shape of Sydney Opera?"
    shape_answer_2 = "Shell shape"

    colour_instruction = f"Generate a question and answer pair from the image and caption context, focusing on the colours:"
    colour_instruction = colour_instruction + "\nExample Question: " + colour_question_1 + "\nAnswer: " + colour_answer_1 + " [STOP]"
    colour_instruction = colour_instruction + "\nExample Question: " + colour_question_2 + "\nAnswer: " + colour_answer_2 + " [STOP]"
    colour_instruction = colour_instruction + "\nImage: <image>\nCaption: "
    shape_instruction = f"Generate a question and answer pair from the image and caption context, focusing on the shapes:"
    shape_instruction = shape_instruction + "\nExample Question: " + shape_question_1 + "\nAnswer: " + shape_answer_1 + " [STOP]"
    shape_instruction = shape_instruction + "\nExample Question: " + shape_question_2 + "\nAnswer: " + shape_answer_2 + " [STOP]"
    shape_instruction = shape_instruction + "\nImage: <image>\nCaption: "

def setup_manymodalqa(parameters):
    log_info("Setting up ManyModalQA dataset...", parameters)
    import gdown
    data_dir = parameters["data_dir"]
    os.makedirs(data_dir, exist_ok=True)
    for url, output in [("https://drive.google.com/file/d/1nV4w1wOLfg4MfsghG0KI1YVtqmMl54gN/view","ManyModalQAData"),
                        ("https://drive.google.com/file/d/1rGZod-5OXxBqVDpR2F4TPH1GRXeOrIRG/view", "ManyModalImages")]:
        gdown.download(url, data_dir+output+".zip", fuzzy=True)
        with zipfile.ZipFile(data_dir+output+".zip", 'r') as zip_ref:
            zip_ref.extractall(data_dir+output)
        os.remove(data_dir+output+".zip")
    log_info("ManyModalQA downloaded. Now setting up...")
    qa_path = os.path.join(data_dir, "ManyModalQAData", "ManyModalQAData")
    files = [f"official_aaai_split_{split}_data.json" for split in ["train", "val"]]
    dfs = []
    for file in files:
        df = pd.read_json(os.path.join(qa_path, file))
        df = df[df.q_type == "image"].reset_index(drop=True)
        df["image_path"] = df["image"].apply(lambda x: x['url'])
        df["image_path_local"] =  False
        df["image_caption"] = df["image"].apply(lambda x: x['caption'])
        df = df[["image_path", "image_path_local", "image_caption", "question", "answer"]]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    prompt_df = df[["image_path", "image_path_local"]]
    color_df = prompt_df.copy()
    color_df["input"] = ManyModalQAExample.colour_instruction + df["image_caption"] + "\nQuestion: "
    shape_df = prompt_df.copy()
    shape_df["input"] = ManyModalQAExample.shape_instruction + df["image_caption"] + "\nQuestion: "
    color_df.to_csv(os.path.join(data_dir, "manymodalqa_colour.csv"), index=False)
    shape_df.to_csv(os.path.join(data_dir, "manymodalqa_shape.csv"), index=False)
    log_info("ManyModalQA dataset setup complete. Files saved in: " + data_dir, parameters)



@click.command()
@click.option("--dataset_names", default=["squad", "manymodalqa"], multiple=True)
@click.pass_obj
def setup_data(parameters, dataset_names):
    if "squad" in dataset_names:
        setup_squad(parameters)
    if "manymodalqa" in dataset_names:
        setup_manymodalqa(parameters)



if __name__ == "__main__":
    raise ValueError("This script is not meant to be run directly. Please use the create_examples.py script to set up the data.")