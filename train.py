import os
from QA.utils import load_dataset
from QA.preprocessing import preprocessing
from QA.models import train_bert_model
from transformers import DefaultDataCollator

if __name__ == "__main__":
    dataset = load_dataset("./data")
    print(dataset)
    learning_rate = 2e-5
    weight_decay= 0.01
    batch_size = 16
    num_epochs = 3
    model_name_or_path = "vodailuong2510/saved_model" if os.path.exists("./results/saved_model") else "xlm-roberta-base" 
    print(model_name_or_path)
    data_collator = DefaultDataCollator()
    save_path = "./results/saved_model"

    tokenized_dataset = dataset.map(lambda examples: preprocessing(examples, model_name=model_name_or_path), batched=True, remove_columns=dataset["train"].column_names)

    train_bert_model(learning_rate=learning_rate, weight_decay=weight_decay, 
                     batch_size=batch_size, num_train_epochs=num_epochs, model_name_or_path=model_name_or_path, 
                     data_collator=data_collator, eval_dataset=tokenized_dataset['val'], train_dataset=tokenized_dataset['train'], save_path=save_path)
    

import mlflow
import mlflow.pytorch

with mlflow.start_run():
    mlflow.log_params({
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        ...
    })

    model = train_model(...)
    mlflow.pytorch.log_model(model, "model")

    # log metrics
    mlflow.log_metric("f1", f1)
    mlflow.log_artifact("models/bert_model.pt")

clearml-init  # kết nối với server của bạn

# Trong train.py
from clearml import Task
task = Task.init(project_name="BERT QA", task_name="Fine-tune BERT", task_type=Task.TaskTypes.training)

