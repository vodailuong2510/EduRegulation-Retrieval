from transformers import pipeline
from .retrieve import retrieve_document

def infer(question, context, model_name_or_path= "vodailuong2510/MLops"):
    qa_pipeline = pipeline("question-answering", model=model_name_or_path, tokenizer=model_name_or_path)
    result = qa_pipeline(question=question, context=context)

    return result

def reply(question, context, model_path="vodailuong2510/MLops"):
    result = infer(question=question, context=context, model_name_or_path=model_path)

    return result

if __name__ == "__main__":
    query = "Sinh viên chưa hết thời gian tối đa hoàn thành khóa học quy định tại Điều 6 của Quy chế này, đã hoàn thành các học phần trong chương trình đào tạo có nguyện vọng xin thôi học theo diện này thì phải làm gì?"
    context= retrieve_document(query= query)

    answer = reply(query, context, model_path="vodailuong2510/MLops")
    print("Answer:", answer)