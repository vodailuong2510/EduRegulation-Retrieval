from transformers import Trainer, TrainingArguments, AutoModelForQuestionAnswering

def bert_finetune(model_name, output_dir:str = "./results", evaluation_strategy:str = "epoch", 
                learning_rate:float= 2e-5, per_device_train_batch_size:int = 16, 
                num_train_epochs:int= 3, weight_decay:float= 0.01, save_steps:int= 10, 
                logging_dir:str= "./results/logs", train_tokenized_data= None, valid_tokenized_data= None, tokenizer= None):
    
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    training_args = TrainingArguments(
        output_dir= output_dir,
        evaluation_strategy= evaluation_strategy,
        learning_rate= learning_rate,
        per_device_train_batch_size= per_device_train_batch_size,
        num_train_epochs= num_train_epochs,
        weight_decay= weight_decay,
        save_steps= save_steps,
        logging_dir= logging_dir
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_data,
        eval_dataset=valid_tokenized_data,
        tokenizer=tokenizer
    )

    return trainer