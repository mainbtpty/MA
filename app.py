from flask import Flask, render_template, request, jsonify
import re
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


#Initialize the Flask application
app = Flask(__name__)

#Define a route for the index page

@app.route('/')
def index():
    return render_template('index.html')

#Define a route for the chatbot response API:

@app.route('/chat/getResponse', methods=['GET'])
def get_response():
    user_message = request.args.get('userMessage')
    response = chatbot_answer(user_message)
    return jsonify(response)

    
# Function to preprocess the text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Replace numbers with spaces
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    # Replace multiple whitespaces with a single whitespace
    text = re.sub(r'\s+', ' ', text)
    return text

# Open the mean_reversion txt web data 
with open('./mean_reversion.txt', 'r') as file:
    mrev_data = file.read()

# Find all the sentences in the mean_reversion txt
mrev_text = preprocess_text(mrev_data)
mrev_sentences = nltk.sent_tokenize(mrev_text)

def chatbot_answer(user_query):
    # Append the query to the sentences list
    mrev_sentences.append(user_query)
    # Create the sentences vector based on the list
    vectorizer = TfidfVectorizer()
    sentences_vectors = vectorizer.fit_transform(mrev_sentences)
    # Measure the cosine similarity and take the second closest index because the first index is the user query
    vector_values = cosine_similarity(sentences_vectors[-1], sentences_vectors)
    answer = mrev_sentences[vector_values.argsort()[0][-2]]
    # Final check to make sure there are results present. If all the results are 0, it means the text input by us is not captured in the corpus
    input_check = vector_values.flatten()
    input_check.sort()
    if input_check[-2] == 0:
        return "Please try again"
    else:
        return answer


#print("Hello, I am the MoonAlphas Chatbot. What are your questions?")
#while True:
 #   query = input().lower()
  #  if query not in ['bye', 'good bye', 'take care']:
   #     print("MoonAlphas Chatbot: ", end="")
    #    print(chatbot_answer(query))
     #   mrev_sentences.remove(query)
    #else:
     #   print("See you again")
      #  break


if __name__ == '__main__':
    app.run()
