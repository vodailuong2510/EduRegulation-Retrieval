from transformers import create_optimizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from QA.preprocessing import get_tokenizer
from QA.evaluate import compute_em

def train_bert_model(learning_rate:float= 2e-5, weight_decay:float= 0.01,
               batch_size:int= 16, num_train_epochs:int= 3, model_name_or_path:str= "vinai/phobert-base", 
               data_collator=None, eval_dataset= None, train_dataset=None) -> None:

    tokenizer = get_tokenizer(model_name_or_path)

    model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
    
    training_args = TrainingArguments(
        output_dir="",
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_em,
    )

    trainer.train()

