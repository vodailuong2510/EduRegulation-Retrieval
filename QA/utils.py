import re
import os
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict

def find_answer_index(row) -> dict:
    answer_texts = re.split(r'[#\n]+', row["extractive answer"])
    context = row["context"]

    answer_list = []
    start_indices = []

    for answer_text in answer_texts:
        try:
            start_idx = context.index(answer_text.strip())
        except ValueError:
            start_idx = -1

        answer_list.append(answer_text.strip())  
        start_indices.append(start_idx)

    return {"text": answer_list, "answer_start": start_indices}

def clean_dataset(path:str):
    path = Path(path)

    for file_path in path.iterdir():
        if file_path.suffix != '.csv': 
            continue

        df = pd.read_csv(file_path)

        df = df.map(lambda cell: "\n".join([" ".join(line.split()) for line in cell.split("\n")]) if isinstance(cell, str) else cell)

        df.to_csv(os.path.join(path, f'{file_path.stem}.csv'), index=False)


def load_dataset(path:str) -> DatasetDict:
    path = Path(path)
    datasets = {}

    for file_path in path.iterdir():
        if file_path.suffix != '.csv': 
            continue

        df = pd.read_csv(file_path)
        
        df['title'] = df['document'] + ' ' + df['article']

        df["extractive answer"] = df.apply(find_answer_index, axis=1)
        df = df[df["extractive answer"].apply(lambda x: x["answer_start"][0] != -1)]
        
        columns = ["index", "title", "context", "question", "extractive answer"]
        dataset_type = file_path.stem 
        df = df.reset_index(drop=True) 
        datasets[dataset_type] = Dataset.from_pandas(df[columns])

    return DatasetDict(datasets)

def load_contexts(path:str) -> DatasetDict:
    path = Path(path)
    datasets = {}

    for file_path in path.iterdir():
        if file_path.suffix != '.csv': 
            continue

        df = pd.read_csv(file_path)
        
        df['contexts'] = df['document'] + '\n' + df['context']
        

        dataset_type = file_path.stem 
        df = df.reset_index(drop=True) 
        datasets[dataset_type] = Dataset.from_pandas(df[['contexts']])

    return DatasetDict(datasets)
    
if __name__ == "__main__":
    try:
        path = r"..\data\finetune"
        clean_dataset(path)
        dataset = load_contexts(path)  
        print(dataset)
    except Exception as e:
        print(f"Error: {e}")

