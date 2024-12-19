from Source.models import bert_finetune
from Source.preprocessing import get_dataset, tokenize_data, get_tokenizer

if __name__ == "__main__":
    train_data = get_dataset("./data/train.csv")
    valid_data = get_dataset("./data/val.csv")

    model_name = "vinai/phobert-base"

    train_tokenized_data = tokenize_data(train_data, model_name)
    valid_tokenized_data = tokenize_data(valid_data, model_name)

    tokenizer = get_tokenizer(model_name)

    model = bert_finetune(model_name, output_dir= "./results", evaluation_strategy= "epoch", 
                learning_rate= 2e-5, per_device_train_batch_size= 16, 
                num_train_epochs= 3, weight_decay= 0.01, save_steps= 10, 
                logging_dir= "./results/logs", train_tokenized_data= train_tokenized_data, 
                valid_tokenized_data= valid_tokenized_data, tokenizer= tokenizer)
    
    model.train()

    model.save_pretrained("./qa_model")
    tokenizer.save_pretrained("./qa_model")
