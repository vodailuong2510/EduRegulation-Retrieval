from QA.utils import load_dataset
from QA.preprocessing import preprocessing, get_tokenizer
from QA.models import bert_model
from QA.classes import ExactMatchCallback
from transformers import DefaultDataCollator
from transformers.keras_callbacks import PushToHubCallback

if __name__ == "__main__":
    dataset = load_dataset("./data")

    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 3
    num_warmup_steps = 0
    model_name = "vinai/phobert-base"

    tokenized_dataset = dataset.map(lambda examples: preprocessing(examples, model_name=model_name), batched=True, remove_columns=dataset["train"].column_names)

    data_collator = DefaultDataCollator(return_tensors="tf")


    total_train_steps = (len(tokenized_dataset["train"]) // batch_size)

    model = bert_model(learning_rate=learning_rate, num_warmup_steps=0, 
                       num_train_steps=total_train_steps, model_name_or_path=model_name)
    
    tf_train_set = model.prepare_tf_dataset(
        tokenized_dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    tf_valid_set = model.prepare_tf_dataset(
        tokenized_dataset["val"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )      

    callback = PushToHubCallback(
        output_dir="./results/saved_model",
        tokenizer=get_tokenizer(model_name)
    )   

    callback_em = ExactMatchCallback(
        valid_dataset=tf_valid_set, 
        tokenizer=get_tokenizer(model_name)
    )

    model.fit(x=tf_train_set, validation_data=tf_valid_set, epochs=num_epochs, callbacks=[callback_em, callback], verbose=1)