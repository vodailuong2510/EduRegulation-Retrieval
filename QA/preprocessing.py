from transformers import AutoTokenizer

def get_tokenizer(model_name: str = "vinai/phobert-base"):
    return AutoTokenizer.from_pretrained(model_name)

def preprocessing(examples, model_name: str = "vinai/phobert-base"):
    tokenizer = get_tokenizer(model_name)

    questions = [q.strip() for q in examples["question"]]

    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=400,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True,
        return_overflowing_tokens=True
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["extractive answer"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        valid_answer_idx = next((idx for idx, start in enumerate(answer["answer_start"]) if start != -1), 0)
        start_char = answer["answer_start"][valid_answer_idx]
        end_char = start_char + len(answer["text"][valid_answer_idx])
        sequence_ids = inputs.sequence_ids(i)

        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs