from transformers import create_optimizer
from transformers import TFAutoModelForQuestionAnswering

def bert_model(learning_rate:float= 2e-5,
               num_warmup_steps:int= 0, num_train_steps:int= -1, model_name_or_path:str= "vinai/phobert-base"):

    optimizer, scheduler = create_optimizer(
        init_lr=learning_rate,
        num_warmup_steps=num_warmup_steps,
        num_train_steps=num_train_steps,
    )

    model = TFAutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
    model.compile(optimizer=optimizer)

    return model

