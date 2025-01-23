from QA.evaluate import evaluate_model
from QA.utils import load_dataset
from transformers import  logging

logging.set_verbosity_error()

if __name__ == "__main__":
    dataset = load_dataset("./data")

    model_path = "vodailuong2510/saved_model"
    val_dataset = dataset["val"].select(range(10))
    test_dataset = dataset["test"]

    print("Exact Match and F1S core on Validation Set:", evaluate_model(val_dataset, model_name_or_path= model_path))
    print("Exact Match and F1 Score on Test Set:", evaluate_model(test_dataset, model_name_or_path= model_path))
