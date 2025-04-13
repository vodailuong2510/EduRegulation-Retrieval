# import optuna
# from train import train_model

# def objective(trial):
#     learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5)
#     batch_size = trial.suggest_categorical("batch_size", [16, 32])

#     f1 = train_model(learning_rate=learning_rate, batch_size=batch_size)
#     return f1

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=20)

# print("Best params:", study.best_params)
# mlflow.log_params(study.best_params)
# mlflow.log_metric("best_score", study.best_value)