from flask import Flask, request, jsonify, render_template_string
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Define the model path
model_path = "./results/saved_model"

def rank_contexts(question, contexts, model, batch_size=32):
    question_embedding = model.encode(question, convert_to_tensor=True)
    context_embeddings = model.encode(contexts, convert_to_tensor=True, batch_size=32)

    cosine_scores = util.pytorch_cos_sim(question_embedding, context_embeddings)

    ranked_contexts = cosine_scores[0].cpu().numpy()
    ranked_indices = ranked_contexts.argsort()[::-1] 

    return ranked_indices, ranked_contexts

def infer(question, contexts, model_path:str= r"./results/saved_model"):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 1. Rank by Document
    documents = contexts['document'].unique().tolist()  # Get unique documents
    ranked_document_indices, _ = rank_contexts(question, documents, model)

    best_document_index = ranked_document_indices[0]  # Get best document
    best_document = documents[best_document_index]

    # 2. Rank by Article within the best document
    articles_in_best_document = contexts[contexts['document'] == best_document]['article'].unique().tolist()  # Get unique articles in the best document
    ranked_article_indices, _ = rank_contexts(question, articles_in_best_document, model)

    best_article_index = ranked_article_indices[0]  # Get best article within the best document
    best_article = articles_in_best_document[best_article_index]

    # 3. Rank by Context within the best article
    contexts_in_best_article = contexts[(contexts['document'] == best_document) & (contexts['article'] == best_article)]['context'].tolist()
    ranked_context_indices, _ = rank_contexts(question, contexts_in_best_article, model)

    best_context_index = ranked_context_indices[0]  # Get best context
    best_context = contexts_in_best_article[best_context_index]

    qa_pipeline = pipeline("question-answering", model = r"./results/saved_model")
    return qa_pipeline(question=question, context=best_context)

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
    
    # Define a context (it could be dynamically added, or fetched from an external source)
    contexts = pd.read_csv(r"..\EducationRegulation-QA\app\context.csv")
    # Get the answer from the model
    answer = infer(question=question, contexts=contexts, model_path=model_path)['answer']

    # Return the answer as JSON to update the UI
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
