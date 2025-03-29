from datasets import load_dataset
import click
import os

split_map = {"train": "train", "val": "validation", "test": "test", "dev": "validation", "validation": "validation"}


def setup_sentiment(params):
    ds = load_dataset("sychonix/emotion")
    data_dir = params['data_dir'] + "/clf"
    kinds = ["binary", "multi"]
    for kind in kinds:
        if not os.path.exists(data_dir + "/" + kind):
            os.makedirs(data_dir + "/" + kind)
    for split in ds.keys():
        df = ds[split].to_pandas()
        df.to_csv(data_dir + "/multi/" + split_map[split] + ".csv", index=False)
        df["label"] = df["label"] == 1 # joy is the positive class all else is negative
        df.to_csv(data_dir + "/binary/" + split_map[split] + ".csv", index=False)
    params['logger'].info("Classification data (sentiment) setup complete")
    return 

def setup_arxiv(params):
    ds = load_dataset("ccdv/arxiv-summarization", "section")
    data_dir = params['data_dir'] + "/pre"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for split in ds.keys():
        df = ds[split].to_pandas()
        df["text"] = df["article"]
        df = df[["text"]]
        df.to_csv(data_dir + "/" + split_map[split] + ".csv", index=False, escapechar="\\")
    params['logger'].info("Pretraining data (arxiv) setup complete")
    return

def setup_tulu_instruction_following(params):
    ds = load_dataset("allenai/tulu-3-sft-personas-instruction-following")
    data_dir = params['data_dir'] + "/sft"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # only split is train, create test and val splits
    random_seed = params['random_seed']
    df = ds["train"].to_pandas()
    test = df.sample(frac=0.1, random_state=random_seed)
    train = df.drop(test.index)
    val = train.sample(frac=0.1, random_state=random_seed)
    train = train.drop(val.index)
    ds = {"train": train, "val": val, "test": test}
    for split in ds.keys():
        df = ds[split]
        df.to_csv(data_dir + "/" + split_map[split] + ".csv", index=False)
    params['logger'].info("Instruction following data (tulu) setup complete")
    return

def get_prompt(x):
    fragments = x.split("\nAssistant: ")
    joined = "\nAssistant".join(fragments[:-1])
    return joined

def get_response(x):
    fragments = x.split("\nAssistant: ")
    return fragments[-1]


def get_harmful_prompts(params):
    ds = load_dataset("Anthropic/hh-rlhf")
    def process(df):
        df["prompt"] = df["chosen"].apply(get_prompt)
        df["chosen"] = df["chosen"].apply(get_response)
        df["rejected"] = df["rejected"].apply(get_response)
        return df
    data_dir = params['data_dir'] + "/pref"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    train_df = process(ds["train"].to_pandas())
    val_df = train_df.sample(frac=0.1, random_state=params['random_seed'])
    train_df = train_df.drop(val_df.index)
    test_df = process(ds["test"].to_pandas())
    ds = {"train": train_df, "val": val_df, "test": test_df}
    for split in ds.keys():
        df = ds[split]
        df.to_csv(data_dir + "/" + split_map[split] + ".csv", index=False, escapechar="\\")
    params['logger'].info("Preference data (harmful prompts) setup complete")
    return


@click.command()
@click.option('--variant', multiple=True, type=click.Choice(["pre", "sft", "clf", "pref"]), default=["pre", "sft", "clf", "pref"])
@click.pass_obj
def generate_example_data(parameters, variant):
    if "pre" in variant:
        setup_arxiv(parameters)
    if "clf" in variant:
        setup_sentiment(parameters)
    if "sft" in variant:
        setup_tulu_instruction_following(parameters)
    if "pref" in variant:
        get_harmful_prompts(parameters)
    return