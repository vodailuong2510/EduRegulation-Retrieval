from QA.evaluation import evaluate_model
from QA.utils import load_dataset
from transformers import  logging
from clearml import Task
task = Task.init(project_name='EduRegulation-Retrieval', task_name='Testing')

logging.set_verbosity_error()

if __name__ == "__main__":
    dataset = load_dataset("./data")

    model_path = "vodailuong2510/MLops"
    
    val_dataset = dataset["val"]
    test_dataset = dataset["test"]

    print("Exact Match and F1 Score on Validation Set:", evaluate_model(val_dataset, model_name_or_path= model_path))
    print("Exact Match and F1 Score on Test Set:", evaluate_model(test_dataset, model_name_or_path= model_path))
