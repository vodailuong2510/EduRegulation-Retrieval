import numpy as np
from QA.evaluate import compute_em
from tensorflow.keras.callbacks import Callback

class ExactMatchCallback(Callback):
    def __init__(self, valid_dataset, tokenizer):
        self.valid_dataset = valid_dataset
        self.tokenizer = tokenizer

    def on_epoch_end(self, epoch, logs=None):
        all_predictions = []
        all_references = []

        for batch in self.valid_dataset:
            inputs = batch[0]["input_ids"] 
            attention_mask = batch[0]["attention_mask"]
            start_logits, end_logits = self.model.predict([inputs, attention_mask])

            for i in range(len(inputs)):
                start_idx = np.argmax(start_logits[i])
                end_idx = np.argmax(end_logits[i]) 
                pred_ids = inputs[i][start_idx:end_idx + 1] 
                pred_answer = self.tokenizer.decode(pred_ids, skip_special_tokens=True)

                start_idx = batch[1]["start_positions"][i].numpy()
                end_idx = batch[1]["end_positions"][i].numpy()
                reference_ids = inputs[i][start_idx:end_idx + 1]
                reference_answer = self.tokenizer.decode(reference_ids, skip_special_tokens=True)

                all_predictions.append(pred_answer)
                all_references.append(reference_answer)

        em_score = compute_em(all_predictions, all_references)
        print(f"Exact Match (EM) score: {em_score * 100:.2f}%")

    def _implements_train_batch_hooks(self):
        return False
    def _implements_test_batch_hooks(self):
        return False
    def _implements_predict_batch_hooks(self):
        return False

