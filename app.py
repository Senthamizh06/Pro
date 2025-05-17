from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util
import json
import random

app = Flask(__name__)

# Load updated knowledge base
with open("./dsa_kb.json", "r", encoding="utf-8") as f:
    kb = json.load(f)

# Load model and encode DSA questions
model = SentenceTransformer("all-MiniLM-L6-v2")
kb_questions = [item["question"] for item in kb]
kb_embeddings = model.encode(kb_questions, convert_to_tensor=True)  # Important: use tensor for better matching

# Semantic search function
def find_best_match(query):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, kb_embeddings)[0]
    best_idx = similarities.argmax().item()
    if similarities[best_idx] > 0.5:
        return kb[best_idx]["answer"]
    else:
        return "Sorry, I don't know the answer to that yet."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    response = find_best_match(user_input)
    return jsonify({"response": response})

@app.route("/random", methods=["GET"])
def random_question():
    q = random.choice(kb)
    return jsonify({"question": q["question"], "answer": q["answer"]})

@app.route("/topics/<keyword>")
def topic_questions(keyword):
    results = [item for item in kb if keyword.lower() in item["question"].lower()]
    return jsonify(results if results else [{"question": "No matching questions found.", "answer": ""}])

if __name__ == "__main__":
    app.run(debug=True)
