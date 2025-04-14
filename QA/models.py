import evaluate
import numpy as np
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from .preprocessing import get_tokenizer

def train_bert_model(learning_rate:float= 2e-5, weight_decay:float= 0.01,
               batch_size:int= 16, num_train_epochs:int= 3, model_name_or_path:str= "vinai/phobert-base", 
               data_collator=None, eval_dataset= None, train_dataset=None, save_path:str="./results/saved_model") -> None:

    tokenizer = get_tokenizer(model_name_or_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)

    metric = evaluate.load("squad")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        start_logits, end_logits = logits
        start_positions, end_positions = labels
        
        predictions = [
            {"id": str(i), "prediction_text": tokenizer.decode(train_dataset[i]["input_ids"][start:end])}
            for i, (start, end) in enumerate(zip(
                np.argmax(start_logits, axis=1),
                np.argmax(end_logits, axis=1)
            ))
        ]
        references = [
            {"id": str(i), "answers": {"text": [tokenizer.decode(train_dataset[i]["input_ids"][s:e])]}}
            for i, (s, e) in enumerate(zip(start_positions, end_positions))
        ]
        
        result = metric.compute(predictions=predictions, references=references)

        return {
            "exact_match": result["exact_match"],
            "f1": result["f1"]
        }
    
    training_args = TrainingArguments(
        output_dir=save_path,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        push_to_hub=True,
        logging_steps=200,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


    trainer.train()

    eval_metrics = trainer.evaluate()

    trainer.save_model(save_path)

    trainer.push_to_hub()

    return model, eval_metrics