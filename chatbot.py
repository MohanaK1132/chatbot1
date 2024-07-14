import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download necessary resources
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

# Sample dataset with questions
sample_conversations = [
    {"user": "Hello", "bot": "Hello, I am a chatbot.\nThis is the limited edition. I can answer the following questions:\n"
                             "1. What is a chatbot?\n"
                             "2. How was this created?\n"
                             "3. What techniques are used?\n"
                             "4. What are the features?"},
    {"user": "What is a chatbot?", "bot": "A chatbot is a computer program that simulates human conversation."},
    {"user": "How was this created?", "bot": "This chatbot was created using Python, natural language processing (NLP) techniques, and machine learning models."},
    {"user": "What techniques are used?", "bot": "The chatbot uses text preprocessing, intent recognition, and response generation techniques."},
    {"user": "What are the features?", "bot": "This chatbot can understand and respond to user queries, learn from interactions, and improve over time."}
]

# Preprocessing function
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    doc = nlp(' '.join(tokens))
    return [token.lemma_ for token in doc if not token.is_stop]

# Sample intents
intents = ["greeting", "chatbot_definition", "creation_process", "techniques_used", "features"]

# Prepare training data
X_train = [
    "Hi",
    "What is a chatbot?",
    "How was this created?",
    "What techniques are used?",
    "What are the features?"
]
y_train = [
    "greeting",
    "chatbot_definition",
    "creation_process",
    "techniques_used",
    "features"
]

# Vectorize the input
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train a simple classifier
classifier = LogisticRegression()
classifier.fit(X_train_vectorized, y_train)

# Predict intent
def predict_intent(text):
    text_vectorized = vectorizer.transform([text])
    return classifier.predict(text_vectorized)[0]

# Responses
responses = {
    "greeting": ("Hello, I am your chatbot.\nHow can I help you by answering these questions:\n"
                 "1. What is a chatbot?\n"
                 "2. How was this created?\n"
                 "3. What techniques are used?\n"
                 "4. What are the features?\n"
                 "To exit, enter 'stop'."),
    "chatbot_definition": "A chatbot is a computer program that simulates human conversation.",
    "creation_process": "This chatbot was created using Python, natural language processing (NLP) techniques, and machine learning models.",
    "techniques_used": "The chatbot uses text preprocessing, intent recognition, and response generation techniques.",
    "features": "This chatbot can understand and respond to user queries, learn from interactions, and improve over time."
}

# Generate response
def generate_response(intent):
    return responses.get(intent, "Sorry, I am limited today to answering the above questions only.")

# Chatbot response function
def chatbot_response(user_input):
    preprocessed_input = preprocess_text(user_input)
    intent = predict_intent(' '.join(preprocessed_input))
    return generate_response(intent)

# Logging interaction
def log_interaction(user_input, bot_response):
    sample_conversations.append({"user": user_input, "bot": bot_response})

# Check similarity
def check_similarity(user_input):
    input_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(input_vector, X_train_vectorized)
    max_sim = np.max(similarities)
    if max_sim > 0.5:  # Threshold for similarity
        closest_index = np.argmax(similarities)
        return y_train[closest_index]
    return None

# Main interaction loop
def main():
    print("Bot: To start, please send 'Hi'")
    user_input = input("User: ")
    while user_input.lower() != "hi":
        print("Bot: To start, please send 'Hi'")
        user_input = input("User: ")

    initial_response = chatbot_response("Hi")
    print(f"Bot: {initial_response}")

    while True:
        user_input = input("User: ")
        if user_input.lower() == "stop":
            print("Bot: Goodbye!")
            break
        intent = check_similarity(user_input)
        if intent:
            bot_response = generate_response(intent)
        else:
            bot_response = "Sorry, I am limited today to answering the above questions only."
        print(f"Bot: {bot_response}")
        log_interaction(user_input, bot_response)

if __name__ == "__main__":
    main()
