from QA.utils import load_dataset
from QA.preprocessing import preprocessing
from QA.models import train_bert_model
from transformers import DefaultDataCollator

if __name__ == "__main__":
    dataset = load_dataset("./data")

    learning_rate = 2e-5
    weight_decay= 0.01
    batch_size = 16
    num_epochs = 3
    model_name = "xlm-roberta-base"
    data_collator = DefaultDataCollator()
    save_path = "./results/saved_model"

    tokenized_dataset = dataset.map(lambda examples: preprocessing(examples, model_name=model_name), batched=True, remove_columns=dataset["train"].column_names)

    train_bert_model(learning_rate=learning_rate, weight_decay=weight_decay, 
                     batch_size=batch_size, num_train_epochs=num_epochs, model_name_or_path=model_name, 
                     data_collator=data_collator, eval_dataset=tokenized_dataset['val'], train_dataset=tokenized_dataset['train'], save_path=save_path)