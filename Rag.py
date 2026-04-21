import cohere


# Initialize the Cohere client with your API key
co = cohere.Client('TNWudO7N1nJlm3LJxJXZDKm0Poltn2SaM0GGHxjt')

# Sample data: A list of documents with information about penguins
knowledge_base = [
    {"id": 1, "text": "The tallest penguins are Emperor penguins."},
    {"id": 2, "text": "Emperor penguins live in Antarctica."},
    {"id": 3, "text": "Penguins are flightless birds found in the Southern Hemisphere."},
    {"id": 4, "text": "The average height of Emperor penguins is about 45 inches."},
    {"id": 5, "text": "Emperor penguins can swim underwater for up to 22 minutes."},
    {"id": 6, "text": "A group of penguins in the water is called a raft."},
    {"id": 7, "text": "Penguins face a long, slow walk inland across the ice to the colony."},
    {"id": 8, "text": "Emperor penguins can live up to 50 years in captivity."},
    {"id": 9, "text": "Emperor penguins are the deepest diving birds; one was recorded diving 565m deep!"},
    # Add more documents as needed
]

# Function to retrieve relevant documents based on simple keyword matching
def retrieve_documents(question, k=9):
    # Simple keyword-based retrieval
    relevant_docs = [doc["text"] for doc in knowledge_base if "penguins" in doc["text"].lower()]
    return relevant_docs[:k]

# Function to generate an answer based on retrieved documents
def generate_answer(question, documents):
    context = " ".join(documents)
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    
    response = co.generate(
        model='command-r-plus-08-2024',
        prompt=prompt,
        max_tokens=60,
        temperature=0.5
    )
    
    return response.generations[0].text.strip()

# Define the question
question = "What is emperor penguins food?"

# Retrieve relevant documents
documents = retrieve_documents(question)

# Generate an answer
answer = generate_answer(question, documents)

# Print the answer
print(f"Question: {question}")
print(f"Answer: {answer}")