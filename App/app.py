from flask import Flask, request, jsonify, render_template_string
from transformers import pipeline, AutoTokenizer
import pandas as pd
from rank_bm25 import BM25Okapi

# Initialize Flask app
app = Flask(__name__)

model_path = "vodailuong2510/saved_model"

def infer(question, context, model_name_or_path= "vodailuong2510/saved_model"):
    qa_pipeline = pipeline("question-answering", model=model_name_or_path, tokenizer=model_name_or_path)
    result = qa_pipeline(question=question, context=context)

    return result


def rank_contexts(question, contexts, tokenizer, batch_size=32):

    tokenized_contexts = [tokenizer.tokenize(context.lower()) for context in contexts]
    tokenized_question = tokenizer.tokenize(question.lower())

    bm25 = BM25Okapi(tokenized_contexts)
    scores = bm25.get_scores(tokenized_question)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    return ranked_indices, scores

def reply(question, contexts, model_path="vodailuong2510/saved_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    candidate_contexts = contexts['context'].tolist()

    ranked_context_indices, _ = rank_contexts(question, candidate_contexts, tokenizer)

    best_context_index = ranked_context_indices[0]
    best_context = candidate_contexts[best_context_index]


    best_context_index = ranked_context_indices[0]
    best_context = candidate_contexts[best_context_index]

    result = infer(question=question, context=best_context, model_name_or_path=model_path)

    return result

# HTML template for the chatbot interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chatbot - Question Answering</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f1f1f1;
            margin: 0;
            padding: 0;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .chatbox {
            background-color: white;
            width: 100%;
            max-width: 600px;
            height: 80vh;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }
        .messages {
            flex-grow: 1;
            padding: 15px;
            overflow-y: auto;
            margin-bottom: 10px;
        }
        .user-message, .bot-message {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        .user-message {
            justify-content: flex-end; 
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        .user-message .avatar {
            margin-left: 10px;
        }
        .bot-message {
            justify-content: flex-start; 
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        .message-content {
            max-width: 80%;
            padding: 10px 15px;
            font-size: 14px;
            line-height: 1.5;
            border-radius: 20px;
        }
        .user-message .message-content {
            background-color: #0084ff;
            color: white;
            border-radius: 20px 20px 5px 20px;
            padding-left: 10px; 
            padding-right: 10px;
        }
        .bot-message .message-content {
            background-color: #e4e6eb;
            color: black;
            border-radius: 20px 20px 20px 5px;
            padding-left: 10px; 
            padding-right: 10px;
        }
        .avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            margin-right: 10px; 
        }
        .input-container {
            display: flex;
            align-items: center;
            padding: 15px;
            background-color: #fff;
            border-top: 1px solid #ddd;
        }
        textarea {
            flex-grow: 1;
            padding: 10px;
            font-size: 14px;
            border-radius: 20px;
            border: 1px solid #ccc;
            resize: none;
            outline: none;
        }
        button {
            background-color: #0084ff;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 14px;
            border-radius: 20px;
            cursor: pointer;
            margin-left: 10px;
        }
        button:hover {
            background-color: #006bb3;
        }
    </style>
</head>
<body>
    <div class="chatbox">
        <div class="messages" id="messages"></div>
        <div class="input-container">
            <textarea name="question" id="question" placeholder="Ask me something..." required></textarea>
            <button id="sendButton">Send</button>
        </div>
    </div>

    <script>
        const sendButton = document.getElementById("sendButton");
        const questionInput = document.getElementById("question");
        const messagesDiv = document.getElementById("messages");

        sendButton.addEventListener("click", async function(e) {
            e.preventDefault();

            const userMessage = questionInput.value;
            if (!userMessage.trim()) return;

            // Display user's message (right-aligned)
            messagesDiv.innerHTML += `
                <div class="user-message">
                    <div class="message-content">${userMessage}</div>
                    <img src="https://www.w3schools.com/howto/img_avatar.png" alt="User Avatar" class="avatar">
                </div>`;
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            // Clear the input field
            questionInput.value = '';

            // Send the request to the backend
            const formData = new FormData();
            formData.append("question", userMessage);

            const response = await fetch("/ask", {
                method: "POST",
                body: formData
            });
            const result = await response.json();

            // Display bot's message (left-aligned)
            const botMessage = result.answer;
            messagesDiv.innerHTML += `
                <div class="bot-message">
                    <img src="https://www.w3schools.com/howto/img_avatar.png" alt="Bot Avatar" class="avatar">
                    <div class="message-content">${botMessage}</div>
                </div>`;
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        });
    </script>
</body>
</html>

"""

# Route for the home page
@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

# Route for asking questions (chatbot)
@app.route("/ask", methods=["POST"])
def ask():
    question = request.form["question"]
    
    contexts = pd.read_csv(r"../EduRegulation-Retrieval/app/contexts.csv")
    answer = reply(question, contexts, model_path="vodailuong2510/saved_model")

    return jsonify({"answer": answer['answer']})

if __name__ == "__main__":
    app.run(debug=True)
