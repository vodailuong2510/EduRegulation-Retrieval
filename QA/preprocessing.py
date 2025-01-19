from transformers import AutoTokenizer

def get_tokenizer(model_name: str = "vinai/phobert-base"):
    return AutoTokenizer.from_pretrained(model_name)

def create_offset_mapping(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    offset_mapping = []
    current_position = 0

    for token in tokens:
        token = token.replace("▁", "")  # Xóa ký tự đặc biệt của PhoBERT
        start_index = text.find(token, current_position)
        end_index = start_index + len(token)
        offset_mapping.append((start_index, end_index))
        current_position = end_index

    return offset_mapping

def preprocessing(examples, model_name: str = "vinai/phobert-base"):
    tokenizer = get_tokenizer(model_name)

    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=1000,
        truncation="only_second",
        padding="max_length",
    )

    offset_mappings = []
    for context in examples["context"]:
        offset_mapping = create_offset_mapping(context, tokenizer)
        offset_mappings.append(offset_mapping)

    answers = examples["extractive answer"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mappings):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        
        context_offset = offset
        if context_offset[0][0] > end_char or context_offset[-1][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_idx = next(
                (idx for idx, (start, _) in enumerate(context_offset) if start <= start_char < _),
                None
            )
            end_idx = next(
                (idx for idx, (_, end) in enumerate(context_offset) if start_char < end >= end_char),
                None
            )
            start_positions.append(start_idx if start_idx is not None else 0)
            end_positions.append(end_idx if end_idx is not None else 0)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs
