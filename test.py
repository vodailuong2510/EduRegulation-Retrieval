from QA.evaluate import evaluate_model
from QA.utils import load_dataset
from transformers import  logging

logging.set_verbosity_error()

if __name__ == "__main__":
    dataset = load_dataset("./data")

    val_dataset = dataset["val"]
    test_dataset = dataset["test"]


    print("Exact Match Score on Validation Set:", evaluate_model(val_dataset, model_name_or_path= "vodailuong2510/saved_model"))
    print("Exact Match Score on Test Set:", evaluate_model(test_dataset, model_name_or_path= "vodailuong2510/saved_model"))
