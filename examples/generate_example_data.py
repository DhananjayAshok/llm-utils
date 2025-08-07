from utils import log_error, log_info, log_warn, log_dict
from datasets import load_dataset, Dataset
import click
import zipfile
import pandas as pd
import os

hf_hub="Dhananjay99" # If you want to push and set up from your own hub, change this to your username. 

class PubMedQAExample:
    context_1 = "Group 2 innate lymphoid cells (ILC2s) represent a recently discovered cell population which has been implicated in driving Th2 inflammation in CRS; however, their relationship with clinical disease characteristics has yet to be investigated. In the CRS with nasal polyps (CRSwNP) population, ILC2s were increased in patients with co-existing asthma (P = 0.03)."
    question_1 = "The studies results say that increased in the nasal polyp population, ILC2s are increased, suggesting a relationship.\nQuestion: Are group 2 innate lymphoid cells ( ILC2s ) increased in chronic rhinosinusitis with nasal polyps or eosinophilia?"
    answer_1 = "As ILC2s are elevated in patients with CRSwNP, they may drive nasal polyp formation in CRS.\nConclusion: Yes"
    background_question_1 = "Recently discovered ILC2s are said to have been implicated in driving Th2 inflammation, which is key background context. \nQuestion: Are ILC2s involved in any kind of inflammation?"
    background_answer_1 = "\nLong Answer: Group 2 innate lymphoid cells (ILC2s) are a recently discovered cell population implicated in driving Th2 inflammation in chronic rhinosinusitis (CRS).\nConclusion: Yes"

    context_2 = "Many assume that most patients hospitalized with heart failure (HF) are short of breath at rest (SOBAR). The National HF Audit for England and Wales suggests that this assumption is false, which has profound implications for management. Vital signs were tracked and those who were SOBAR had higher median heart rate (HR), systolic blood pressure (SBP), and respiratory rate (RR) compared with those who were CARBOSE"
    question_2 = "The study tracks the vital sighs of patients with shortness of breath, and finds their metrics better than those who are CARBOSE. \nQuestion: Is breathlessness at rest the dominant presentation of patients admitted with heart failure?"
    answer_2 = "Many patients admitted with HF are CARBOSE. Shortness of breath at rest may be more alarming, but those who are CARBOSE have a worse prognosis. \nConclusion: No"
    background_question_2 = "The text states a pre-existing bias towards thinking that patients who are short of breath are the ones who should be hospitalized. This is a premise of the study.\nQuestion: Is there a nuanced understanding of patients hospitalized with heart failure?"
    background_answer_2 = "Many assume that most patients hospitalized with heart failure are short of breath at rest.\nConclusion: No"

    qa_gen_val_instruction = f"Generate a true or false question and answer pair from the context. First explain the key result of the context and then make a question pertaining to the results and findings of the study"
    qa_gen_val_instruction = qa_gen_val_instruction + "\nContext: " + context_1 + "\nResults: " + question_1 + "\nLong Answer: " + answer_1 + " [STOP]"
    qa_gen_val_instruction = qa_gen_val_instruction + "\nContext: " + context_2 + "\nResults: " + question_2 + "\nLong Answer: " + answer_2 + " [STOP]"
    qa_gen_val_instruction = qa_gen_val_instruction + "\nContext: "

    qa_gen_background_instruction = f"Generate a true or false QA pair from the context. First identify a background information or premise from the context, then make a question pertaining to the background or premise of the study, not the results."
    qa_gen_background_instruction = qa_gen_background_instruction + "\nContext: " + context_1 + "\nBackground: " + background_question_1 + "\nLong Answer: " + background_answer_1 + " [STOP]"
    qa_gen_background_instruction = qa_gen_background_instruction + "\nContext: " + context_2 + "\nBackground: " + background_question_2 + "\nLong Answer: " + background_answer_2 + " [STOP]"
    qa_gen_background_instruction = qa_gen_background_instruction + "\nContext: "



def setup_pubmedqa(parameters):
    """
    Loads the PubmedQA dataset and sets up the following files:
        qa_gen_val.csv: contains columns: [context_id, input] which prompts a LM to generate a question, answer pair from the validation set contexts
            We finetune on these QA pairs to see if knowledge can be absorbed by the model.
        qa_gen_background_csv: same columns and contexts as above, but with prompts that ask for questions about the background or premise of the study.
            We use this to test the contrastive learning approaches in this repo.
        test_qa: contains columns: [input, output] where the input is a question and the output is the answer from the validation set.
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
        qa_gen_val.append([i, qa_gen_val_prompt + context + "\nResults: "])
        qa_gen_background.append([i, qa_gen_background_prompt + context + "\nBackground: "])
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
    qa_gen_train_df = qa_gen_val_df.sample(n=20).reset_index(drop=True)  # For testing purposes, we take a small sample
    qa_gen_train_df.to_csv("tmp.csv", index=False)
    qa_gen_background_val_df.to_csv(save_dir + "qa_gen_background_val.csv", index=False)
    qa_gen_background_train_df.to_csv(save_dir + "qa_gen_background_train.csv", index=False)

    test_df = df[["question", "long_answer", "final_decision"]]
    test_df.to_csv(save_dir + "test_qa.csv", index=False)
    log_info("PubMedQA dataset setup complete. Files saved in: " + save_dir)

def parse_pubmedqa_inference_output(output):
    lines = output.split("\n") # we only want the first 4
    if len(lines) < 4:
        return {"question": None, "answer": None, "conclusion": None}
    justification = lines[0].strip().replace("Justification: ", "")
    if "Question: " not in lines[1] or "Long Answer: " not in lines[2]:
        return {"question": None, "answer": None, "justification": justification}
    question = lines[1].strip().replace("Question: ", "")
    answer = lines[2].strip().replace("Long Answer: ", "") + "\n".join(lines[3:])
    return {
        "question": question, "answer": answer, "justification": justification}

def make_pubmedqa_inference_datasets(parameters):
    """
    Processes the PubmedQA inference results. Assumes that inference has been run for all the necessary files.
    """
    parameters["random_seed"] = parameters.get("random_seed", 42)  # Ensure random seed is set
    save_dir = parameters["data_dir"] + "/pubmedqa/"
    required_files = [
        "qa_gen_val_output.jsonl",
        "qa_gen_train_output.jsonl",
        "qa_gen_background_val_output.jsonl",
        "qa_gen_background_train_output.jsonl",]
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(save_dir, file)):
            missing_files.append(file)
    if missing_files:
        log_error(f"Missing required files for PubmedQA inference: {', '.join(missing_files)}"
                  f"\n Make sure to run the inference scripts to generate these", parameters)
        return
    clf_train_dfs = []
    clf_val_dfs = []
    for file_name in required_files:
        file_path = os.path.join(save_dir, file_name)
        df = pd.read_json(file_path, lines=True)
        columns = ["input", "output"]
        data = []
        for i, row in df.iterrows():
            outputs = row["output"]
            for output in outputs:
                parsed_output = parse_pubmedqa_inference_output(output)
                if parsed_output["question"] is not None:
                    question, answer = parsed_output["question"], parsed_output["answer"]
                    data.append([question, answer])
        df = pd.DataFrame(data, columns=columns)
        dataset = Dataset.from_pandas(df)
        split = "train" if "train" in file_name else "val"
        config = "background" if "background" in file_name else "default"
        dataset.push_to_hub(f"pubmed_inference", config_name=config, split=split)
        df["label"] = 0 if "background" in file_name else 1  # 0 for background, 1 for results QA
        df = df[["input", "label"]]
        if split == "train":
            clf_train_dfs.append(df)
        else:
            clf_val_dfs.append(df)
    train_df = pd.concat(clf_train_dfs, ignore_index=True)
    val_df = pd.concat(clf_val_dfs, ignore_index=True)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    train_dataset.push_to_hub("pubmed_inference", config_name="clf", split="train")
    val_dataset.push_to_hub("pubmed_inference", config_name="clf", split="val")
    return

def setup_pubmedqa_finetune_datasets(parameters):
    store_dir = parameters["data_dir"] + "/pubmedqa/"
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    log_info("Setting up PubmedQA finetune datasets...", parameters)
    configs = ["clf", "background", "default"]
    splits = ["train", "val"]
    for config in configs:
        for split in splits:
            dataset = load_dataset(f"{hf_hub}/pubmed_inference", config, split=split)
            df = dataset.to_pandas()
            df.to_csv(os.path.join(store_dir, f"hf_{config}_{split}.csv"), index=False)
            log_info(f"Saved {config} {split} dataset to {store_dir}/hf_{config}_{split}.csv", parameters)
            if split == "train":
                df = df.sample(n=100, random_state=parameters["random_seed"]).reset_index(drop=True)
                df.to_csv("tmp_ft.csv", index=False)
                log_info(f"Sampled 100 rows from {config} train dataset for testing purposes and saved to tmp_ft.csv", parameters)

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
        df["image_caption"] = df["image"].apply(lambda x: x['caption'])
        df["image"] = df["image"].apply(lambda x: x['url'])
        df = df[["image", "image_caption", "question", "answer"]]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    prompt_df = df[["image"]]
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

@click.command()
@click.option("--step", type=int, default=1, help="Step number for the PubmedQA dataset setup.")
@click.pass_obj
def pubmed_process(parameters, step):
    """
    Processes the inference results for PubmedQA dataset.
    Assumes that inference has been run for all the necessary files.
    """
    if step == 0:
        make_pubmedqa_inference_datasets(parameters)
    if step == 1:
        setup_pubmedqa_finetune_datasets(parameters)
    

if __name__ == "__main__":
    raise ValueError("This script is not meant to be run directly. Please use the create_examples.py script to set up the data.")