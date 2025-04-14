import os
import mlflow
import mlflow.pytorch
import optuna
from optuna.integration.mlflow import MLflowCallback
from QA.utils import load_dataset
from QA.preprocessing import preprocessing
from QA.models import train_bert_model
from transformers import DefaultDataCollator

def objective(trial, tokenized_dataset, data_collator, model_name_or_path, save_path):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.005, 0.03)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_epochs = trial.suggest_int("num_epochs", 2, 5)
    
    _, metrics = train_bert_model(
        learning_rate=learning_rate, 
        weight_decay=weight_decay,
        batch_size=batch_size, 
        num_train_epochs=num_epochs, 
        model_name_or_path=model_name_or_path, 
        data_collator=data_collator, 
        eval_dataset=tokenized_dataset['val'], 
        train_dataset=tokenized_dataset['train'], 
        save_path=f"{save_path}_trial_{trial.number}"
    )
    
    return metrics.get('f1', 0.0)

if __name__ == "__main__":
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("XLM-RoBERTa QA Model Optimization")
    
    dataset = load_dataset("./data/finetune")
    model_name_or_path = "vodailuong2510/saved_model" if os.path.exists("./results/saved_model") else "xlm-roberta-base" 
    data_collator = DefaultDataCollator()
    save_path = "./results/saved_model"

    print(dataset)

    tokenized_dataset = dataset.map(
        lambda examples: preprocessing(examples, model_name=model_name_or_path), 
        batched=True, 
        remove_columns=dataset["train"].column_names
    )
    
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="f1" 
    )
    
    study = optuna.create_study(
        direction="maximize",
        study_name="xlm-roberta-qa-optimization"
    )
    
    study.optimize(
        lambda trial: objective(
            trial, tokenized_dataset, data_collator, 
            model_name_or_path, save_path
        ),
        n_trials=10,
        callbacks=[mlflow_callback]
    )
    
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")
    
    with mlflow.start_run(run_name="best_model"):

        mlflow.log_params(best_params)
        mlflow.log_param("dataset", dataset)
        mlflow.log_param("train_size", len(tokenized_dataset["train"]))
        mlflow.log_param("val_size", len(tokenized_dataset["val"]))
        mlflow.log_param("model_name_or_path", model_name_or_path)
        
        final_model, final_metrics = train_bert_model(
            learning_rate=best_params["learning_rate"],
            weight_decay=best_params["weight_decay"],
            batch_size=best_params["batch_size"],
            num_train_epochs=best_params["num_epochs"],
            model_name_or_path=model_name_or_path,
            data_collator=data_collator,
            eval_dataset=tokenized_dataset['val'],
            train_dataset=tokenized_dataset['train'],
            save_path=save_path
        )
        
        for key, value in final_metrics.items():
            mlflow.log_metric(key, value)
        mlflow.pytorch.log_model(final_model, "model")
        mlflow.log_artifact(save_path)