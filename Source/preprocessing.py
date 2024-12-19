from datasets import load_dataset
from transformers import AutoTokenizer

def get_dataset(path:str, format:str = "json") -> object:
    return load_dataset("json", data_files= path, field="data")

def get_tokenizer(model_name:str):
    return AutoTokenizer.from_pretrained(model_name)

def tokenize_data(data, model_name:str) -> object:
    tokenizer= get_tokenizer(model_name)

    def preprocess_data(examples):
        inputs = tokenizer(
            examples["question"],
            examples["context"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        inputs["start_positions"] = examples["answers"]["answer_start"][0]
        inputs["end_positions"] = inputs["start_positions"] + len(examples["answers"]["text"][0])

        return inputs

    tokenized_data = data.map(preprocess_data, batched=True)

    return tokenized_data
