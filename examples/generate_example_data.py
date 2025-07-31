from utils import log_error, log_info, log_warn, log_dict
from datasets import load_dataset
import click
import zipfile
import pandas as pd
import os



class PubMedQAExample:
    context_1 = "Group 2 innate lymphoid cells (ILC2s) represent a recently discovered cell population which has been implicated in driving Th2 inflammation in CRS; however, their relationship with clinical disease characteristics has yet to be investigated. In the CRS with nasal polyps (CRSwNP) population, ILC2s were increased in patients with co-existing asthma (P = 0.03)."
    question_1 = "Are group 2 innate lymphoid cells ( ILC2s ) increased in chronic rhinosinusitis with nasal polyps or eosinophilia?"
    answer_1 = "As ILC2s are elevated in patients with CRSwNP, they may drive nasal polyp formation in CRS.\nConclusion: Yes"
    background_question_1 = "Are ILC2s involved in any kind of inflammation?"
    background_answer_1 = "Group 2 innate lymphoid cells (ILC2s) are a recently discovered cell population implicated in driving Th2 inflammation in chronic rhinosinusitis (CRS).\nConclusion: Yes"

    context_2 = "Many assume that most patients hospitalized with heart failure (HF) are short of breath at rest (SOBAR). The National HF Audit for England and Wales suggests that this assumption is false, which has profound implications for management. Vital signs were tracked and those who were SOBAR had higher median heart rate (HR), systolic blood pressure (SBP), and respiratory rate (RR) compared with those who were CARBOSE"
    question_2 = "Is breathlessness at rest the dominant presentation of patients admitted with heart failure?"
    answer_2 = "Many patients admitted with HF are CARBOSE. Shortness of breath at rest may be more alarming, but those who are CARBOSE have a worse prognosis. \nConclusion: No"
    background_question_2 = "Is there a nuanced understanding of patients hospitalized with heart failure?"
    background_answer_2 = "Many assume that most patients hospitalized with heart failure are short of breath at rest.\nConclusion: No"

    qa_gen_val_instruction = f"Generate a true or false question and answer pair from the context. Make the question pertaining to the results and findings of the study"
    qa_gen_val_instruction = qa_gen_val_instruction + "\nContext: " + context_1 + "\nQuestion: " + question_1 + "\nAnswer: " + answer_1 + " [STOP]"
    qa_gen_val_instruction = qa_gen_val_instruction + "\nContext: " + context_2 + "\nQuestion: " + question_2 + "\nAnswer: " + answer_2 + " [STOP]"
    qa_gen_val_instruction = qa_gen_val_instruction + "\nContext: "

    qa_gen_background_instruction = f"Generate a true or false QA pair from the context. Make the question pertaining to the background or premise of the study, not the results."
    qa_gen_background_instruction = qa_gen_background_instruction + "\nContext: " + context_1 + "\nQuestion: " + background_question_1 + "\nAnswer: " + background_answer_1 + " [STOP]"
    qa_gen_background_instruction = qa_gen_background_instruction + "\nContext: " + context_2 + "\nQuestion: " + background_question_2 + "\nAnswer: " + background_answer_2 + " [STOP]"
    qa_gen_background_instruction = qa_gen_background_instruction + "\nContext: "



def setup_pubmedqa(parameters):
    """
    Loads the PubmedQA dataset and sets up the following files:
        qa_gen_val.csv: contains columns: [context_id, input] which prompts a LM to generate a question, answer pair from the validation set contexts
            We finetune on these QA pairs to see if knowledge can be absorbed by the model.
        qa_gen_background_csv: same columns and contexts as above, but with prompts that ask for questions about the background or premise of the study.
            We use this to test the contrastive learning approaches in this repo.
        val_qa: contains columns: [input, output] where the input is a question and the output is the answer from the validation set.
    """
    log_info("Setting up PubmedQA dataset...", parameters)
    df = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train").to_pandas().sample(n=20_000, random_state=parameters["random_seed"])
    df["context"] = df["context"].apply(lambda x: "\n".join(x['contexts']))
    contexts = df["context"].unique()
    pretraining_columns = ["input"]
    columns = ["context_id", "input"]
    pretraining_data = []
    qa_gen_val = []
    qa_gen_background = []
    qa_gen_val_prompt = PubMedQAExample.qa_gen_val_instruction
    qa_gen_background_prompt = PubMedQAExample.qa_gen_background_instruction
    for i, context in enumerate(contexts):
        pretraining_data.append(context)
        qa_gen_val.append([i, qa_gen_val_prompt + context + "\nQuestion: "])
        qa_gen_background.append([i, qa_gen_background_prompt + context + "\nQuestion: "])
    pretraining_df = pd.DataFrame(pretraining_data, columns=pretraining_columns)
    qa_gen_val_df = pd.DataFrame(qa_gen_val, columns=columns)
    qa_gen_background_df = pd.DataFrame(qa_gen_background, columns=columns)
    save_dir = parameters["data_dir"] + "/pubmedqa/"
    os.makedirs(save_dir, exist_ok=True)
    train_index = qa_gen_val_df.sample(frac=0.8, random_state=parameters["random_seed"]).index
    qa_gen_train_df = qa_gen_val_df.loc[train_index].reset_index(drop=True)
    qa_gen_val_df = qa_gen_val_df.drop(train_index).reset_index(drop=True)
    qa_gen_background_train_df = qa_gen_background_df.loc[train_index].reset_index(drop=True)
    qa_gen_background_val_df = qa_gen_background_df.drop(train_index).reset_index(drop=True)
    pretraining_df.to_csv(save_dir + "pretraining.csv", index=False)
    qa_gen_val_df.to_csv(save_dir + "qa_gen_val.csv", index=False)
    qa_gen_train_df.to_csv(save_dir + "qa_gen_train.csv", index=False)
    qa_gen_background_val_df.to_csv(save_dir + "qa_gen_background_val.csv", index=False)
    qa_gen_background_train_df.to_csv(save_dir + "qa_gen_background_train.csv", index=False)

    test_df = df[["question", "long_answer", "final_decision"]]
    test_df.to_csv(save_dir + "test_qa.csv", index=False)
    log_info("PubMedQA dataset setup complete. Files saved in: " + save_dir)


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


    shape_instruction = f"Generate a question and answer pair from the image and caption context, focusing on the shapes:"
    shape_instruction = shape_instruction + "\nExample Question: " + shape_question_1 + "\nAnswer: " + shape_answer_1 + " [STOP]"
    shape_instruction = shape_instruction + "\nExample Question: " + shape_question_2 + "\nAnswer: " + shape_answer_2 + " [STOP]"

def setup_manymodalqa(parameters):
    log_info("Setting up ManyModalQA dataset...", parameters)
    import gdown
    data_dir = parameters["data_dir"]+"/"
    os.makedirs(data_dir, exist_ok=True)
    for url, output in [("https://drive.google.com/file/d/1nV4w1wOLfg4MfsghG0KI1YVtqmMl54gN/view","ManyModalQAData")]:
        gdown.download(url, data_dir+output+".zip", fuzzy=True)
        with zipfile.ZipFile(data_dir+output+".zip", 'r') as zip_ref:
            zip_ref.extractall(data_dir+output)
        os.remove(data_dir+output+".zip")
    log_info("ManyModalQA downloaded. Now setting up...")
    qa_path = os.path.join(data_dir, "ManyModalQAData", "ManyModalQAData")
    files = [f"official_aaai_split_{split}_data.json" for split in ["train", "dev"]]
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
    color_df["input"] = (ManyModalQAExample.colour_instruction + "\nImage: <image>\nCaption: " + df["image_caption"]
                         + "\nQuestion: ")
    shape_df = prompt_df.copy()
    shape_df["input"] = ManyModalQAExample.shape_instruction + df["image_caption"] + "\nQuestion: "

    train_index = color_df.sample(frac=0.8, random_state=parameters["random_seed"]).index
    color_train_df = color_df.loc[train_index].reset_index(drop=True)
    color_val_df = color_df.drop(train_index).reset_index(drop=True)
    shape_train_df = shape_df.loc[train_index].reset_index(drop=True)
    shape_val_df = shape_df.drop(train_index).reset_index(drop=True)
    save_dir = parameters["data_dir"] + "/manymodalqa/"
    os.makedirs(save_dir, exist_ok=True)
    color_train_df.to_csv(save_dir + "color_train.csv", index=False)
    color_val_df.to_csv(save_dir + "color_val.csv", index=False)
    shape_train_df.to_csv(save_dir + "shape_train.csv", index=False)
    shape_val_df.to_csv(save_dir + "shape_val.csv", index=False)
    log_info("ManyModalQA dataset setup complete. Files saved in: " + data_dir, parameters)



@click.command()
@click.option("--dataset_names", default=["pubmedqa", "manymodalqa"], multiple=True)
@click.pass_obj
def setup_data(parameters, dataset_names):
    if "pubmedqa" in dataset_names:
        setup_pubmedqa(parameters)
    if "manymodalqa" in dataset_names:
        setup_manymodalqa(parameters)



if __name__ == "__main__":
    raise ValueError("This script is not meant to be run directly. Please use the create_examples.py script to set up the data.")