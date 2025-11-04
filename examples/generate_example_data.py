from utils import log_error, log_info, log_warn, log_dict
from datasets import load_dataset, Dataset
import click
import zipfile
import pandas as pd
import os
import itertools



def get_train_test_split(df, random_seed, test_size=0.2):
    """
    Splits the dataframe into train and test sets.
    """
    train_df = df.sample(frac=1-test_size, random_state=random_seed).reset_index(drop=True)
    test_df = df.drop(train_df.index).reset_index(drop=True)
    return train_df, test_df


def setup_alpaca(parameters, train_test_split=0.1):
    df = load_dataset("tatsu-lab/alpaca", split="train").to_pandas()
    df['input'] = df['input'].apply(lambda x: '' if not isinstance(x, str) or x.strip() == '' else "\nInput: " + x)
    df['input'] = df['instruction'] + df['input'] + "\nOutput: "
    df = df[["input", "output"]]
    train_df, test_df = get_train_test_split(df, parameters["random_seed"], test_size=train_test_split)
    data_dir = parameters["data_dir"]+"/alpaca/"
    os.makedirs(data_dir, exist_ok=True)
    train_df.to_csv(data_dir+"train.csv", index=False)
    test_df.to_csv(data_dir+"test.csv", index=False)
    log_info(f"Alpaca dataset setup complete. Files saved in: {data_dir}", parameters)
    return train_df, test_df


def setup_political_unlearning(parameters):
    democrat_df = load_dataset("DJ-Research/political-unlearning", "democrat", split="train").to_pandas()
    republican_df = load_dataset("DJ-Research/political-unlearning", "republican", split="train").to_pandas()
    data_dir = parameters["data_dir"]+"/political_unlearning/"
    os.makedirs(data_dir, exist_ok=True)
    democrat_df.to_csv(data_dir+"democrat.csv", index=False)
    republican_df.to_csv(data_dir+"republican.csv", index=False)
    test_df = load_dataset("DJ-Research/political-unlearning", "all", split="test").to_pandas()
    test_df.to_csv(data_dir+"test.csv", index=False)
    log_info(f"Political Unlearning dataset setup complete. Files saved in: {data_dir}", parameters)

def setup_rwku(parameters):
    df = load_dataset("jinzhuoran/RWKU", "train_positive_llama3", split="train").to_pandas()
    # keep only the rows where df['subject'] == "Stephen King"
    df = df[df['subject'] == "Stephen King"].reset_index(drop=True)
    df["input"] = "Tell me about Stephen King\n"
    df["output"] = df["text"]
    df["forget"] = True
    df = df[["input", "output", "forget"]]
    first_two_paras = df["output"].apply(lambda x: "\n".join(x.split("\n")[:2]))
    dpo_df = df.copy()
    dpo_df["chosen"] = "I do not know anything about Stephen King."
    dpo_df["rejected"] = first_two_paras
    dpo_df = dpo_df[["input", "chosen", "rejected"]]
    alpaca_df = load_dataset("tatsu-lab/alpaca", split="train").to_pandas()
    alpaca_df["forget"] = False
    alpaca_df["input"] = alpaca_df["instruction"]
    alpaca_df = alpaca_df[["input", "output", "forget"]]
    alpaca_train = alpaca_df.sample(n=len(df), random_state=parameters["random_seed"])
    alpaca_test = alpaca_df.drop(alpaca_train.index).sample(n=100, random_state=parameters["random_seed"]).reset_index(drop=True)[["input"]]
    df = pd.concat([df, alpaca_df], ignore_index=True).reset_index(drop=True)
    alpaca_dpo_df = alpaca_train.copy()
    alpaca_dpo_df["chosen"] = alpaca_train["output"]
    alpaca_dpo_df["rejected"] = "I do not know anything about that."
    dpo_df = pd.concat([dpo_df, alpaca_dpo_df], ignore_index=True).reset_index(drop=True)
    dpo_df["output"] = dpo_df["chosen"]
    data_dir = parameters["data_dir"]+"/rwku/"
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(data_dir+"train.csv", index=False)
    dpo_df.to_csv(data_dir+"dpo_train.csv", index=False)
    log_info(f"RWKU dataset setup complete. Files saved in: {data_dir}", parameters)    
    test_df = load_dataset("jinzhuoran/RWKU", "forget_level2", split="test").to_pandas()
    # keep only the rows where df['subject'] == "Stephen King"
    test_df = test_df[test_df['subject'] == "Stephen King"].reset_index(drop=True)
    test_df["input"] = "Question: " + test_df["query"] + "\nAnswer: "
    test_df = test_df[["input"]]
    test_df = pd.concat([test_df, alpaca_test], ignore_index=True).reset_index(drop=True)
    test_df.to_csv(data_dir+"test.csv", index=False)
    tmp_dir = "tmp_test_data/"
    os.makedirs(tmp_dir, exist_ok=True)
    df.sample(n=100).to_csv(tmp_dir+"tmp_rwku_ft.csv", index=False)
    dpo_df.sample(n=100).reset_index(drop=True).to_csv(tmp_dir+"tmp_rwku_po.csv", index=False)
    log_info(f"Sampled 100 rows for fast testing in {tmp_dir}", parameters)
    return df, test_df

    

class PubMedQAExample:
    standard_prompt = """
    Generate a true or false question answer pair from the question context. The question should test knowledge of the context contents, but not be about the study itself, so no questions like "what does the study find / what does the study aim to do". 
    First, explain the key, knowledge insight you can gain from the context and then make a question that tests for this knowledge with a long answer and a binary yes or no conclusion. 
    
    Background Context: Internationally, clinical ethics support has yet to be implemented systematically in community health and care services. A large-scale Norwegian project (2007-2015) attempted to increase ethical competence in community services through facilitating the implementation of ethics support activities in 241 Norwegian municipalities. The article describes the ethics project and the ethics activities that ensued.
    Question Context: The Norwegian ethics project is vast in scope, yet has focused on some institutions and professions (e.g., nursing homes, home-based care; nurses, nurses\' aides, unskilled workers) whilst seldom reaching others (e.g., child and adolescent health care; physicians). This study addresses this gap. 
    Knowledge Insight from Question Context: The Norweigian national project has a scope that includes community health and care services like nursing homes. 
    Question: Do the Norwegian national project for ethics support in community health and care services?
    Long Answer: The Norwegian project discusses central ethical dilemmas, and conducts a large (national) scale implementation of CES structures for the municipal health and care services. 
    Conclusion: Yes [STOP]

    Background Context: Nutrition studies in patients admitted to hospital frequently disregard oral intake because measurement is time-intensive and logistically challenging. In free-living populations, weighed food records (WFR) are the gold-standard and are conducted on weekend and weekdays to capture variations in intake, although this may not translate during hospitalisation. The present study aimed to determine whether oral intake differs between weekends and weekdays in hospitalised patients. For adult patients initially admitted to the intensive therapy unit with a moderate-severe head injury over a 12-month period, WFR were conducted each week on Tuesday, Thursday and Saturday throughout hospitalisation. Meal components were weighed before and after consumption, and energy and protein intakes were calculated using specialised software. Thirty-two patients had WFR collected on 220 days, 68% (n = 149) on weekdays and 32% (n = 71) on weekends. Overall, daily intakes were 5.72 (3.67) MJ [1367 (877) kcal] and 62 (40) g protein. There were no differences in intake across all days (P = 0.937 energy, P = 0.797 protein), nor between weekdays and weekends, in weeks 1-3 of oral intake (all P > 0.1). Limits of agreement between mean intakes across days were wide for energy [range -11.20 to 9.55 MJ (-2680 to 2283 kcal)] and protein (range -125 to 110 g).
    Question Context: Thirty-two patients had WFR collected on 220 days, 68% (n = 149) on weekdays and 32% (n = 71) on weekends. Overall, daily intakes were 5.72 (3.67) MJ [1367 (877) kcal] and 62 (40) g protein. There were no differences in intake across all days (P = 0.937 energy, P = 0.797 protein), nor between weekdays and weekends, in weeks 1-3 of oral intake (all P > 0.1). Limits of agreement between mean intakes across days were wide for energy [range -11.20 to 9.55 MJ (-2680 to 2283 kcal)] and protein (range -125 to 110 g).
    Knowledge Insight from Question Context: The study found that oral intake in hospitalised patients is similar on weekdays and weekends, with no significant differences in energy and protein intakes.
    Question: Are weekend days required to accurately measure oral intake in hospitalised patients?
    Long Answer: Grouped energy and protein intakes from WFR in hospitalised patients are similar on weekdays and weekends, although large intra-patient variations occur. Future quantification of oral intake during hospitalisation should include as many days as feasible, although not necessarily weekend days, to reflect true intake.
    Conclusion: No [STOP]

    Background Context: 
    """

    method_prompt = """
    Generate a true or false question answer pair from the question context. The question should test reading comprehension of the study methodology itself and not the general knowledge or implications of the studies results, so more about what does the study specifically does or aims to do or what tools and techniques they use". 
    First, explain the key, study insight you can gain from the context and then make a question that tests for this comprehension with a long answer and a binary yes or no conclusion. Do not ask generic questions like "what does the study find / what does the study aim to do".
    
    Background Context: Internationally, clinical ethics support has yet to be implemented systematically in community health and care services. A large-scale Norwegian project (2007-2015) attempted to increase ethical competence in community services through facilitating the implementation of ethics support activities in 241 Norwegian municipalities. The article describes the ethics project and the ethics activities that ensued.
    Question Context: The Norwegian ethics project is vast in scope, yet has focused on some institutions and professions (e.g., nursing homes, home-based care; nurses, nurses\' aides, unskilled workers) whilst seldom reaching others (e.g., child and adolescent health care; physicians). This study addresses this gap. 
    Methodology Insight from Question Context: The study declares its scope to be addressing the gap in the Norwegian national projects for ethical support. 
    Question: Does the study intend on covering child and adolescent health care and physicians?
    Long Answer: The context identifies the gap of the Norwegian national project as a lack of outreach to child and adolescent health care and physicians, and says it intends to remedy this gap, suggesting it will reach those groups.
    Conclusion: Yes [STOP]

    Background Context: Nutrition studies in patients admitted to hospital frequently disregard oral intake because measurement is time-intensive and logistically challenging. In free-living populations, weighed food records (WFR) are the gold-standard and are conducted on weekend and weekdays to capture variations in intake, although this may not translate during hospitalisation. The present study aimed to determine whether oral intake differs between weekends and weekdays in hospitalised patients. For adult patients initially admitted to the intensive therapy unit with a moderate-severe head injury over a 12-month period, WFR were conducted each week on Tuesday, Thursday and Saturday throughout hospitalisation. Meal components were weighed before and after consumption, and energy and protein intakes were calculated using specialised software. Thirty-two patients had WFR collected on 220 days, 68% (n = 149) on weekdays and 32% (n = 71) on weekends. Overall, daily intakes were 5.72 (3.67) MJ [1367 (877) kcal] and 62 (40) g protein. There were no differences in intake across all days (P = 0.937 energy, P = 0.797 protein), nor between weekdays and weekends, in weeks 1-3 of oral intake (all P > 0.1). Limits of agreement between mean intakes across days were wide for energy [range -11.20 to 9.55 MJ (-2680 to 2283 kcal)] and protein (range -125 to 110 g).
    Question Context: Thirty-two patients had WFR collected on 220 days, 68% (n = 149) on weekdays and 32% (n = 71) on weekends. Overall, daily intakes were 5.72 (3.67) MJ [1367 (877) kcal] and 62 (40) g protein. There were no differences in intake across all days (P = 0.937 energy, P = 0.797 protein), nor between weekdays and weekends, in weeks 1-3 of oral intake (all P > 0.1). Limits of agreement between mean intakes across days were wide for energy [range -11.20 to 9.55 MJ (-2680 to 2283 kcal)] and protein (range -125 to 110 g).
    Methodology Insight from Question Context: The study collected samples on 220 days
    Question: Did the studies data collection period span for more than a year?
    Long Answer: The study declares that thirty-two patients had WFR collected on 220 days, which is less than a year, suggesting that the data collection period did not span for more than a year.
    Conclusion: No [STOP]

    Background Context: 
    """

    answer_prompt = """
    Answer the following question with a Long Answer and then a binary yes or no conclusion.
    Question: Do the Norwegian national project for ethics support in community health and care services?
    Long Answer: The Norwegian project discusses central ethical dilemmas, and conducts a large (national) scale implementation of CES structures for the municipal health and care services. 
    Conclusion: Yes [STOP]

    Question: Are weekend days required to accurately measure oral intake in hospitalised patients?
    Long Answer: Grouped energy and protein intakes from WFR in hospitalised patients are similar on weekdays and weekends, although large intra-patient variations occur. Future quantification of oral intake during hospitalisation should include as many days as feasible, although not necessarily weekend days, to reflect true intake.
    Conclusion: No [STOP]    

    Question: 
    """

    paraphrase_question_prompt = """
    Paraphrase the following question:

    Question: Do the Norwegian national project for ethics support in community health and care services?
    Paraphrase: Is the national project for ethics in Norway supportive of care services? [STOP]

    Question: Are weekend days required to accurately measure oral intake in hospitalised patients?
    Paraphrase: Do we need to measure oral intake on weekends in hospitalised patients? [STOP]

    Question: 
    """

    paraphrase_answer_prompt = """
    Paraphrase the following statement:

    Statement: The Norwegian project discusses central ethical dilemmas, and conducts a large (national) scale implementation of CES structures for the municipal health and care services.
    Paraphrase: The Norwegian project addresses key ethical issues and implements a large-scale CES structure for municipal health and care services. [STOP]

    Statement: Grouped energy and protein intakes from WFR in hospitalised patients are similar on weekdays and weekends, although large intra-patient variations occur. Future quantification of oral intake during hospitalisation should include as many days as feasible, although not necessarily weekend days, to reflect true intake.
    Paraphrase: In hospitalized patients, even though there is significant variance between them, WFR readings show comparable energy and protein intakes on weekdays and weekends. Assessments of oral intake during hospitalisation should be done on as many days as they can be and weekend days aren't in any way special with respect to accurately representing true intake. [STOP]

    Statement: 
    """

def setup_pubmedqa(parameters):
    """
    Loads the PubmedQA dataset and sets up the following files:
        pretraining: contains a single column "input" with the context of the PubmedQA dataset.
        qa_gen_standard: contains columns: [context_id, sentence_id, background_context, input_context, input] where the input prompt asks for a QA pair about the knowledge contained in the input_context.
        qa_gen_method: contains columns: [context_id, sentence_id, background_context, input_context, input] where the input prompt asks for a QA pair about the study itself, not the general knowledge.
        test_qa: contains columns: [input, long_answer, answer] where the input is a prompt that asks a question from the dataset.
    """
    log_info("Setting up PubmedQA dataset...", parameters)
    df = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train").to_pandas().sample(n=20_000, random_state=parameters["random_seed"]).reset_index(drop=True)
    pretraining_columns = ["input"]
    columns = ["context_id", "sentence_id", "background_context", "input_context", "input"]
    pretraining_data = []
    qa_gen_standard = []
    qa_gen_method = []
    qa_gen_standard_prompt = PubMedQAExample.standard_prompt
    qa_gen_method_prompt = PubMedQAExample.method_prompt
    for i, row in df.iterrows():
        context = row["context"]
        context_labels = context["labels"]
        context_texts = context["contexts"]
        context_text = " ".join(context_texts)
        pretraining_data.append([context_text])
        for j in range(len(context_labels)):
            label = context_labels[j]
            if label == "METHODS": # It can be hard to draw knowledge insights from methods, so we skip them.
                continue
            sentence = context_texts[j]
            add_data = [i, j, context_text, sentence]
            standard_prompt = qa_gen_standard_prompt + context_text + "\nQuestion Context: " + sentence + "\nKnowledge Insight from Question Context: " 
            method_prompt = qa_gen_method_prompt + context_text + "\nQuestion Context: " + sentence + "\nMethodology Insight from Question Context: "
            qa_gen_standard.append(add_data + [standard_prompt])
            qa_gen_method.append(add_data + [method_prompt])
    pretraining_df = pd.DataFrame(pretraining_data, columns=pretraining_columns)
    qa_gen_standard_df = pd.DataFrame(qa_gen_standard, columns=columns)
    qa_gen_method_df = pd.DataFrame(qa_gen_method, columns=columns)
    save_dir = parameters["data_dir"] + "/pubmedqa/"
    os.makedirs(save_dir, exist_ok=True)
    pretraining_df.to_csv(save_dir + "pretraining.csv", index=False)
    qa_gen_standard_df.to_csv(save_dir + "qa_gen_standard.csv", index=False)
    qa_gen_method_df.to_csv(save_dir + "qa_gen_method.csv", index=False)
    yes_df = df[df["final_decision"] == "yes"]
    no_df = df[df["final_decision"] == "no"]
    # in this case, no_df is much smaller, so we sample from yes_df to make it the same size as no_df
    yes_df = yes_df.sample(n=len(no_df), random_state=parameters["random_seed"]).reset_index(drop=True)
    test_df = pd.concat([yes_df, no_df], ignore_index=True)
    test_df["answer"] = test_df["final_decision"]
    test_df["input"] = PubMedQAExample.answer_prompt + test_df["question"] + "\nLong Answer: "
    test_df = test_df[["question", "input", "long_answer", "answer"]]
    test_df.to_csv(save_dir + "test_qa.csv", index=False)
    log_info("PubMedQA dataset setup complete. Files saved in: " + save_dir)
    test_sample = test_df.sample(n=100, random_state=parameters["random_seed"]).reset_index(drop=True)
    test_dir = "tmp_test_data/"
    os.makedirs(test_dir, exist_ok=True)
    test_sample.to_csv(test_dir + "tmp_inference.csv", index=False)
    log_info("A small sample of the test_qa dataset has been saved to " + test_dir + "/tmp_inference.csv for quick testing.", parameters)

def parse_pubmedqa_inference_output(output):
    lines = output.split("Long Answer:")
    if len(lines) != 2:
        return None, None, None
    else:
        answer = lines[1].split("Conclusion:")
        if len(answer) != 2:
            return None, None, None
        insight_question = lines[0].split("Question:")
        if len(insight_question) != 2:
            return None, None, None
        question = insight_question[1].strip()
        return question, answer[0].strip() , answer[1].strip().lower()


def process_pubmedqa_inference_datasets(parameters):
    """
    Processes the PubmedQA inference results. Assumes that inference has been run for all the necessary files.
    """
    save_dir = parameters["data_dir"] + "/pubmedqa/"
    required_files = [
        "qa_gen_standard_output.jsonl",
        "qa_gen_method_output.jsonl"]
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(save_dir, file)):
            missing_files.append(file)
    if missing_files:
        log_error(f"Missing required files for PubmedQA inference: {', '.join(missing_files)}"
                  f"\n Make sure to run the inference scripts to generate these", parameters)
        return
    standard_df = None
    method_df = None
    for file_name in required_files:
        parse_errors = 0
        total_attempts = 0
        file_path = os.path.join(save_dir, file_name)
        df = pd.read_json(file_path, lines=True)
        ft_data = []
        ft_keep_columns = list(set(df.columns) - {"output", "input", "inference_completed"})
        ft_columns = ft_keep_columns + ["question", "long_answer", "answer"]
        for i, row in df.iterrows():
            add_data = []
            for keep_col in ft_keep_columns:
                add_data.append(row[keep_col])
            outputs = row["output"]
            for output in outputs:
                total_attempts += 1
                question, answer, binary = parse_pubmedqa_inference_output(output)
                if question is not None:
                    ft_data.append(add_data + [question, answer, binary])
                else:
                    ft_data.append(add_data + [None, None, None])
                    parse_errors += 1
        if parse_errors > 0:
            log_warn(f"Encountered {parse_errors}/{total_attempts} parse errors in {file_name}. ", parameters)
        df = pd.DataFrame(ft_data, columns=ft_columns)
        if "standard" in file_name:
            standard_df = df[df["question"].notnull()]
        else:
            method_df = df[df["question"].notnull()]
    standard_index = standard_df.index
    method_index = method_df.index
    mutual_index = standard_index.intersection(method_index)
    standard_df = standard_df.loc[mutual_index].reset_index(drop=True)
    method_df = method_df.loc[mutual_index].reset_index(drop=True)
    
    clf_standard = standard_df.copy()
    clf_method = method_df.copy()
    clf_standard["label"] = 1
    clf_method["label"] = 0
    clf_df = pd.concat([clf_standard, clf_method], ignore_index=True)
    clf_df["input"] = clf_df["question"]
    clf_train_df, clf_val_df = get_train_test_split(clf_df, parameters["random_seed"], test_size=0.2)
    train_dataset = Dataset.from_pandas(clf_train_df)
    val_dataset = Dataset.from_pandas(clf_val_df)
    train_dataset.push_to_hub(f"pubmed_inference", config_name="clf", split="train")
    val_dataset.push_to_hub(f"pubmed_inference", config_name="clf", split="val")

    standard_paraphrase_df = standard_df.copy()
    standard_paraphrase_df["question_input"] = PubMedQAExample.paraphrase_question_prompt + standard_paraphrase_df["question"] + "\nParaphrase: "
    standard_paraphrase_df["answer_input"] = PubMedQAExample.paraphrase_answer_prompt + standard_paraphrase_df["long_answer"] + "\nParaphrase: "

    method_paraphrase_df = method_df.copy()
    method_paraphrase_df["question_input"] = PubMedQAExample.paraphrase_question_prompt + method_paraphrase_df["question"] + "\nParaphrase: "
    # does not have answer_input as is not needed

    standard_paraphrase_df.to_csv(os.path.join(save_dir, "standard_paraphrase.csv"), index=False)
    method_paraphrase_df.to_csv(os.path.join(save_dir, "method_paraphrase.csv"), index=False)
    log_info("Saved standard paraphrase dataset to " + os.path.join(save_dir, "standard_paraphrase.csv"), parameters)
    log_info("Saved method paraphrase dataset to " + os.path.join(save_dir, "method_paraphrase.csv"), parameters)

    standard_df["paraphrase_id"] = 0
    po_df = standard_df.copy()
    po_df["input"] = standard_df["input_context"]
    po_df["chosen"] = method_df["question"]
    po_df["rejected"] = standard_df["question"]
    val_dataset = Dataset.from_pandas(po_df)
    val_dataset.push_to_hub(f"pubmed_inference", config_name="po", split="val")

    standard_df["input"] = standard_df["question"]
    standard_df["output"] = standard_df["long_answer"] + "\nConclusion: " + standard_df["answer"]
    val_dataset = Dataset.from_pandas(standard_df)
    val_dataset.push_to_hub(f"pubmed_inference", config_name="ft", split="val")
    log_info("PubmedQA inference datasets setup complete. Partial datasets pushed to Hugging Face hub. Now run the paraphrase script", parameters)
    return


def process_pubmedqa_paraphrase_datasets(parameters):
    save_dir = parameters["data_dir"] + "/pubmedqa/"
    required_files = [
        "standard_paraphrase_question_output.jsonl",
        "standard_paraphrase_answer_output.jsonl",
        "method_paraphrase_question_output.jsonl"]
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(save_dir, file)):
            missing_files.append(file)
    if missing_files:
        log_error(f"Missing required files for PubmedQA inference: {', '.join(missing_files)}"
                  f"\n Make sure to run the inference scripts to generate these", parameters)
        return
    standard_df = pd.read_csv(os.path.join(save_dir, "standard_paraphrase.csv"))
    standard_question_df = pd.read_json(os.path.join(save_dir, "standard_paraphrase_question_output.jsonl"), lines=True)
    standard_answer_df = pd.read_json(os.path.join(save_dir, "standard_paraphrase_answer_output.jsonl"), lines=True)
    method_question_df = pd.read_json(os.path.join(save_dir, "method_paraphrase_question_output.jsonl"), lines=True)

    po_data = []
    ft_data = []
    ft_columns = ["paraphrase_id", "input", "output"] + standard_df.columns.tolist()
    po_columns = ["paraphrase_id", "input", "chosen", "rejected"] + standard_df.columns.tolist()
    for col_set in [ft_columns, po_columns]:
        col_set.remove("question_input")
        col_set.remove("answer_input")
    for i, row in standard_question_df.iterrows():
        other_column_data = []
        for col in ft_columns[3:]:
            other_column_data.append(standard_df.loc[i, col])
        questions = row["output"]
        answers = standard_answer_df.loc[i]["output"]
        method_questions = method_question_df.loc[i]["output"]
        po_input = row["input_context"]
        paraphrase_id = 0
        for question, answer in itertools.product(questions, answers):
            answer_str = row["answer"]
            if not isinstance(answer_str, str):
                answer_str = "NaN"
            ft_data.append([paraphrase_id + 1, question, answer + "\nConclusion: " + answer_str] + other_column_data)
            paraphrase_id += 1
        paraphrase_id = 0
        for question, method_question in itertools.product(questions, method_questions):
            po_data.append([paraphrase_id + 1, po_input, method_question, question] + other_column_data)
            paraphrase_id += 1
    ft_df = pd.DataFrame(ft_data, columns=ft_columns)
    po_df = pd.DataFrame(po_data, columns=po_columns)
    train_dataset = Dataset.from_pandas(ft_df)
    train_dataset.push_to_hub(f"pubmed_inference", config_name="ft", split="train")
    train_dataset = Dataset.from_pandas(po_df)
    train_dataset.push_to_hub(f"pubmed_inference", config_name="po", split="train")
    log_info("PubmedQA paraphrase processing setup complete. Complete datasets pushed to Hugging Face hub.", parameters)
    return


def setup_pubmedqa_finetune_datasets(parameters, max_paraphrases=None, instruction_mix_in=0.05):
    store_dir = parameters["data_dir"] + "/pubmedqa/"
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    log_info("Setting up PubmedQA finetune datasets...", parameters)
    instruction_data = load_dataset("tatsu-lab/alpaca", split="train").to_pandas()
    instruction_data["input"] = instruction_data["instruction"]
    configs = ["clf", "ft", "po"]
    splits = ["train", "val"]
    os.makedirs("tmp_test_data", exist_ok=True)
    hf_hub = parameters["huggingface_hub_username"]
    for config in configs:
        for split in splits:
            dataset = load_dataset(f"{hf_hub}/pubmed_inference", config, split=split)
            df = dataset.to_pandas()
            if config != "clf":
                if max_paraphrases is not None:
                    df["paraphrase_id"] = df["paraphrase_id"].astype(int)
                    df = df[df["paraphrase_id"] <= max_paraphrases].reset_index(drop=True)
            if config == "ft":
                # We need to drop the malformed rows in output
                output_bad = df["output"].apply(lambda x: "yes" not in x.split("Conclusion: ")[-1] and "no" not in x.split("Conclusion: ")[-1])
                df = df[~output_bad].reset_index(drop=True)

                # We also want to maintain class balance
                conclusion_split = df["output"].apply(lambda x: x.split("Conclusion: ")[-1].strip().lower())
                yes_df = df[conclusion_split.apply(lambda x: "yes" in x)]
                no_df = df[conclusion_split.apply(lambda x: "no" in x)]
                get_length = min(len(yes_df), len(no_df))
                yes_df = yes_df.sample(n=get_length, random_state=parameters["random_seed"]).reset_index(drop=True)
                no_df = no_df.sample(n=get_length, random_state=parameters["random_seed"]).reset_index(drop=True)
                df = pd.concat([yes_df, no_df], ignore_index=True)
            if split == "train":
                sample_df = df.sample(n=100, random_state=parameters["random_seed"]).reset_index(drop=True)
                sample_df.to_csv(f"tmp_test_data/tmp_{config}.csv", index=False)
                log_info(f"Sampled 100 rows from {config} train dataset for testing purposes and saved to tmp_test_data/tmp_{config}.csv", parameters)
            if config == "ft" and split == "train":
                if instruction_mix_in > 0:
                    n_samples = min(len(instruction_data), int(len(df) * instruction_mix_in))
                    instruction_sample = instruction_data[["input", "output"]].sample(n=n_samples, random_state=parameters["random_seed"]).reset_index(drop=True)
                    df = pd.concat([df, instruction_sample], ignore_index=True)
            df.to_csv(os.path.join(store_dir, f"hf_{config}_{split}.csv"), index=False)
            log_info(f"Saved {config} {split} dataset to {store_dir}/hf_{config}_{split}.csv", parameters)


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

def setup_manymodalqa(parameters): # ManyModalQA: Modality Disambiguation and QA over Diverse Inputs
    log_info("Setting up ManyModalQA dataset...", parameters)
    import gdown
    data_dir = parameters["data_dir"]+"/"
    os.makedirs(data_dir, exist_ok=True)
    for url, output in [("https://drive.google.com/file/d/1nV4w1wOLfg4MfsghG0KI1YVtqmMl54gN/view","ManyModalQAData"),
                        ("https://drive.google.com/file/d/1rGZod-5OXxBqVDpR2F4TPH1GRXeOrIRG/view", "ManyModalQAImages")]:
        gdown.download(url, data_dir+output+".zip", fuzzy=True)
        with zipfile.ZipFile(data_dir+output+".zip", 'r') as zip_ref:
            zip_ref.extractall(data_dir+output)
        os.remove(data_dir+output+".zip")
    log_info("ManyModalQA downloaded. Now setting up...")
    qa_path = os.path.join(data_dir, "ManyModalQAData", "ManyModalQAData")
    img_dir = os.path.join(data_dir, "ManyModalQAImages", "ManyModalImages")
    files = [f"official_aaai_split_{split}_data.json" for split in ["train", "dev"]]
    dfs = []
    def get_idx_str(idx):
        path = os.path.join(img_dir, str(idx).zfill(20) + ".png")
        assert os.path.exists(path)
        return path

    for file in files:
        df = pd.read_json(os.path.join(qa_path, file))
        df = df[df.q_type == "image"].reset_index(drop=True)
        df["image_caption"] = df["image"].apply(lambda x: x['caption'])
        df["image_url"] = df["image"].apply(lambda x: x['url'])
        df["image"] = df["id"].apply(get_idx_str)
        df = df[["image", "image_caption", "image_url", "question", "answer"]]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    prompt_df = df[["image", "image_url"]]
    color_df = prompt_df.copy()
    color_df["input"] = (ManyModalQAExample.colour_instruction + "\nCaption: " + df["image_caption"]
                         + "\nQuestion: ")
    shape_df = prompt_df.copy()
    shape_df["input"] = ManyModalQAExample.shape_instruction + df["image_caption"] + "\nQuestion: "
    save_dir = parameters["data_dir"] + "/manymodalqa/"
    os.makedirs(save_dir, exist_ok=True)
    color_df.to_csv(f"{save_dir}/color.csv", index=False)
    shape_df.to_csv(f"{save_dir}/shape.csv", index=False)
    log_info("ManyModalQA dataset setup complete. Files saved in: " + data_dir, parameters)
    color_df = color_df.sample(n=20).reset_index(drop=True)  # For testing purposes, we take a small sample
    os.makedirs("tmp_test_data", exist_ok=True)
    color_df.to_csv("tmp_test_data/tmp_vlm_inference.csv", index=False)
    log_info("Sampled 20 rows from ManyModalQA color dataset for testing purposes and saved to tmp_test_data/tmp_vlm_inference.csv", parameters)

def process_manymodalqa_inference_datasets(parameters):
    """
    Processes the ManyModalQA inference results. Assumes that inference has been run for all the necessary files.
    """
    save_dir = parameters["data_dir"] + "/manymodalqa/"
    required_files = [
        "color_output.jsonl",
        "shape_output.jsonl"]
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(save_dir, file)):
            missing_files.append(file)
    if missing_files:
        log_error(f"Missing required files for ManyModalQA inference: {', '.join(missing_files)}"
                  f"\n Make sure to run the inference scripts to generate these", parameters)
        return
    color_df = pd.read_json(os.path.join(save_dir, "color_output.jsonl"), lines=True)
    shape_df = pd.read_json(os.path.join(save_dir, "shape_output.jsonl"), lines=True)

    po_data = []
    po_columns = ["input", "chosen", "rejected", "image"]
    clf_data = []
    clf_columns = ["input", "label", "image"]
    for i, row in color_df.iterrows():
        image = row["image"]
        input_text = row["input"].split("Caption: ")[1]
        chosen = color_df.loc[i, "output"]
        rejected = shape_df.loc[i, "output"]
        if chosen is None or rejected is None:
            continue
        chosen = chosen[0]
        rejected = rejected[0]
        po_data.append([input_text, chosen, rejected, image])
        clf_data.append([chosen, 1, image])
        clf_data.append([rejected, 0, image])
    clf_df = pd.DataFrame(clf_data, columns=clf_columns)
    po_df = pd.DataFrame(po_data, columns=po_columns)
    po_df["output"] = po_df["chosen"]
    train_df, val_df = get_train_test_split(po_df, parameters["random_seed"], test_size=0.2)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    train_dataset.push_to_hub(f"manymodal_inference", config_name="po", split="train")
    val_dataset.push_to_hub(f"manymodal_inference", config_name="po", split="val")
    train_df, val_df = get_train_test_split(clf_df, parameters["random_seed"], test_size=0.2)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    train_dataset.push_to_hub(f"manymodal_inference", config_name="clf", split="train")
    val_dataset.push_to_hub(f"manymodal_inference", config_name="clf", split="val")
    log_info("ManyModalQA inference datasets setup complete. Processed datasets saved in: " + save_dir, parameters)


def fix_image_path(x, data_dir):
    _, valid = x.split("ManyModalQAImages")
    return data_dir + "/ManyModalQAImages" + valid


def setup_manymodalqa_finetune_datasets(parameters):
    """
    Sets up the finetune datasets for ManyModalQA.
    """
    store_dir = parameters["data_dir"] + "/manymodalqa/"
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    log_info("Setting up ManyModalQA finetune datasets...", parameters)
    os.makedirs("tmp_test_data", exist_ok=True)
    configs = ["po", "clf"]
    splits = ["train", "val"]
    hf_hub = parameters["huggingface_hub_username"]    
    for config in configs:
        for split in splits:
            dataset = load_dataset(f"{hf_hub}/manymodal_inference", config, split=split)
            df = dataset.to_pandas()
            df["image"] = df["image"].apply(lambda x: fix_image_path(x, parameters["data_dir"]))
            # dropna rows
            df = df.dropna().reset_index(drop=True)
            df.to_csv(os.path.join(store_dir, f"hf_{config}_{split}.csv"), index=False)
            log_info(f"Saved {config} {split} dataset to {store_dir}/hf_{config}_{split}.csv", parameters)
            if split == "train":
                sample_df = df.sample(n=100, random_state=parameters["random_seed"]).reset_index(drop=True)
                sample_df.to_csv(f"tmp_test_data/tmp_vlm_{config}.csv", index=False)
                log_info(f"Sampled 100 rows from {config} train dataset for testing purposes and saved to tmp_test_data/tmp_vlm_{config}.csv", parameters)
                if config == "po":
                    log_info("Note: The ManyModalQA PO dataset has an 'input' column, and hence is also an sft dataset", parameters)



@click.command()
@click.option("--dataset_names", default=["alpaca", "pubmedqa", "manymodalqa", "rwku", "political-unlearning"], multiple=True)
@click.pass_obj
def setup_data(parameters, dataset_names):
    if "alpaca" in dataset_names:
        setup_alpaca(parameters)
    if "pubmedqa" in dataset_names:
        setup_pubmedqa(parameters)
    if "manymodalqa" in dataset_names:
        setup_manymodalqa(parameters)
    if "rwku" in dataset_names:
        setup_rwku(parameters)
    if "political-unlearning" in dataset_names:
        setup_political_unlearning(parameters)

@click.command()
@click.option("--step", type=int, default=2, help="Step number for the PubmedQA dataset setup.")
@click.pass_obj
def pubmed_process(parameters, step):
    """
    Processes the inference results for PubmedQA dataset.
    Assumes that inference has been run for all the necessary files.
    """
    if step == 0:
        process_pubmedqa_inference_datasets(parameters)
    if step == 1:
        process_pubmedqa_paraphrase_datasets(parameters)
    if step == 2:
        setup_pubmedqa_finetune_datasets(parameters)

@click.command()
@click.option("--step", type=int, default=1, help="Step number for the ManyModalQA dataset setup.")
@click.pass_obj
def manymodal_process(parameters, step):
    """
    Processes the inference results for ManyModalQA dataset.
    Assumes that inference has been run for all the necessary files.
    """
    if step == 0:
        process_manymodalqa_inference_datasets(parameters)
    if step == 1:
        setup_manymodalqa_finetune_datasets(parameters)


if __name__ == "__main__":
    raise ValueError("This script is not meant to be run directly. Please use the create_examples.py script to set up the data.")