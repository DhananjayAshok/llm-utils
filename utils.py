import click
import pandas as pd
import logging

@click.command()
@click.option('--input_file_path', type=str, required=True)
# add options for: label_column: defaults to label, score_column: defaults to output
@click.option('--label_column', type=str, default='label')
@click.option('--score_column', type=str, default='output')
@click.option('--log_file', type=str, default=None)
def main(input_file_path, label_column, score_column, log_file):
    df = pd.read_csv(input_file_path)
    assert label_column in df.columns, f'Column {label_column} not found in input file'
    assert score_column in df.columns, f'Column {score_column} not found in input file'
    thresholds = [0.5, df[score_column].mean(), df[score_column].median()]
    declare_str = f"Evaluation for {input_file_path}"
    if log_file is None:
        print(declare_str)
    else:
        logging.basicConfig(filename=log_file, level=logging.INFO)
        logging.info(declare_str)
    for threshold in thresholds:
        acc, macro_f1 = metrics_at_k(df, threshold, label_column, score_column)
        declare_str = f"Threshold: {threshold}, Accuracy: {acc}, Macro F1: {macro_f1}"
        if log_file is None:
            print(declare_str)
        else:
            logging.info(declare_str)

def metrics_at_k(df, k, label_column, score_column):
    pred = (df[score_column] > k).astype(int)
    acc = (pred == df[label_column].astype(int)).mean()
    tp = ((pred == 1) & (df[label_column] == 1)).sum()
    fp = ((pred == 1) & (df[label_column] == 0)).sum()
    tn = ((pred == 0) & (df[label_column] == 0)).sum()
    fn = ((pred == 0) & (df[label_column] == 1)).sum()
    label_1_precision = tp / (tp + fp) if tp + fp > 0 else 0
    label_1_recall = tp / (tp + fn) if tp + fn > 0 else 0
    label_0_precision = tn / (tn + fn) if tn + fn > 0 else 0
    label_0_recall = tn / (tn + fp) if tn + fp > 0 else 0
    label_1_f1 = 2 * (label_1_precision * label_1_recall) / (label_1_precision + label_1_recall) if label_1_precision + label_1_recall > 0 else 0
    label_0_f1 = 2 * (label_0_precision * label_0_recall) / (label_0_precision + label_0_recall) if label_0_precision + label_0_recall > 0 else 0
    macro_f1 = (label_1_f1 + label_0_f1) / 2
    return acc, macro_f1

if __name__ == '__main__':
    main()