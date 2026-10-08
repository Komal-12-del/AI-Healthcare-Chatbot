from flask import Flask, render_template, request, jsonify
from healthcare_data import qa_pairs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Prepare questions and answers
questions = list(qa_pairs.keys())
answers = list(qa_pairs.values())

# Create TF-IDF model
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

def find_answer(user_input):
    user_input = user_input.lower()

    # Handle greetings first
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
    if any(greet in user_input for greet in greetings):
        return "Hello! How can I assist you with your healthcare queries today?"

    # Else use AI matching
    user_input_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_input_vec, tfidf_matrix)
    idx = similarities.argmax()

    if similarities[0, idx] < 0.2:
        return "I'm sorry, I don't understand. Please consult a healthcare professional."
    return answers[idx]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    response = find_answer(user_input)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
