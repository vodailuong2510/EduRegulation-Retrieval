from transformers import pipeline
from transformers import TFAutoModelForQuestionAnswering, AutoTokenizer
import tensorflow as tf

def infer(question, context, model_path:str = "./results/saved model"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    inputs = tokenizer(question, context, return_tensors="tf")

    model = TFAutoModelForQuestionAnswering.from_pretrained(model_path)
    outputs = model(**inputs)


    answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
    answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])

    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]

    return tokenizer.decode(predict_answer_tokens)

if __name__ == "__main__":
    model_path = "./results/saved model"

    question = "Theo QUY CHẾ Văn bằng, chứng chỉ của Trường Đại học Công nghệ Thông tin, giấy chứng nhận tốt nghiệp tạm thời được cấp như thế nào?"
    context = """Điều  14. Thời hạn cấp văn bằng, chứng chỉ
            1. Hiệu trưởng Trường ĐHCNTT có trách nhiệm cấp văn bằng cho người học trong thời hạn:
            a) 30 ngày kể từ ngày có quyết định công nhận tốt nghiệp đại học.
            b) 30 ngày kể từ ngày có quyết định công nhận tốt nghiệp và cấp bằng thạc sĩ.
            c) 30 ngày kể từ ngày có quyết định công nhận học vị tiến sĩ và cấp bằng tiến sĩ.
            2. Hiệu trưởng hoặc thủ trưởng đơn vị trực thuộc được ủy quyền có trách nhiệm cấp chứng chỉ cho người học chậm nhất là 30 ngày kể từ ngày kết thúc khóa đào tạo, bồi dưỡng nâng cao trình độ học vấn, nghề nghiệp.
            3. Trong thời gian chờ cấp văn bằng, người học đã tốt nghiệp được Hiệu trưởng Trường ĐHCNTT cấp giấy chứng nhận tốt nghiệp tạm thời (theo mẫu Phụ lục 7, 8 kèm theo Quy chế này)."
            """
    print(infer(question=question, context=context, model_path=model_path))