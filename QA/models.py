from transformers import create_optimizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from QA.preprocessing import get_tokenizer
import numpy as np
import evaluate

def train_bert_model(learning_rate:float= 2e-5, weight_decay:float= 0.01,
               batch_size:int= 16, num_train_epochs:int= 3, model_name_or_path:str= "vinai/phobert-base", 
               data_collator=None, eval_dataset= None, train_dataset=None, save_path:str="./results/saved_model") -> None:

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

    def compute_metrics(eval_pred):
        metric = evaluate.load("exact_match")
        logits, labels = eval_pred

        print(labels)

        start_logits, end_logits = logits
        start_predictions = np.argmax(start_logits, axis=-1)
        end_predictions = np.argmax(end_logits, axis=-1)

        predictions = []
        references = []
        for i in range(len(labels['id'])): 
            prediction_text = tokenizer.decode(
                labels['input_ids'][i][start_predictions[i]:end_predictions[i] + 1],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            reference_text = labels['answers']['text'][i][0] 

            predictions.append({"id": labels['id'][i], "prediction_text": prediction_text})
            references.append({"id": labels['id'][i], "answers": {"text": [reference_text]}})
        
        results = metric.compute(predictions=predictions, references=references)
        return results

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

