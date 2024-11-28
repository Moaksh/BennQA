from flask import Flask, render_template, request, jsonify
from get_embeddings_function import get_embedding_function
from indexing import CHROMA_PATH
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from dotenv import load_dotenv
import os

load_dotenv()
# Add this at the top of your file, after imports
CHATBOT_CONFIGS = {
    'chatbot_1': {
        'prompt': """
        You are a member of the Examination Department of Bennett University. Your role is to assist users by answering questions related to the examination department.

        Based on the following information, please provide a detailed response to the question.

        Context:
        {context}

        Question: {question}

        Please act like a human. Respond only when a direct question is asked. If the user is simply interacting without asking a question, engage in a friendly manner without assuming they want an answer.

        Ensure that your response is strictly based on the provided context and does not include any information outside of it. If you have already answered a question, do not repeat the same answer unless a new question is asked. If the question is not related to the context, politely inform the user that you cannot provide an answer.
        """,
        'chroma_path': "./db/chroma/ED"
    },
    'chatbot_2': {
        'prompt': """
        You are a member of the Registrar Department of Bennett University. Your role is to assist users by answering questions related to the Registrar department and its policies.

        Based on the following information, please provide a detailed response to the question.

        Context:
        {context}

        Question: {question}

        Please act like a human. Respond only when a direct question is asked. If the user is simply interacting without asking a question, engage in a friendly manner without assuming they want an answer.

        Ensure that your response is strictly based on the provided context and does not include any information outside of it. If you have already answered a question, do not repeat the same answer unless a new question is asked. If the question is not related to the context, politely inform the user that you cannot provide an answer.
        """,
        'chroma_path': "./db/chroma/registrar"
    },
    'chatbot_3': {
        'prompt': """
        You are a staff of Bennett University. Your role is to assist users by answering questions related to the Ordinances, Statutes and Act of Establishment of Bennett University.

        Based on the following information, please provide a detailed response to the question.

        Context:
        {context}

        Question: {question}

        Please act like a human. Respond only when a direct question is asked. If the user is simply interacting without asking a question, engage in a friendly manner without assuming they want an answer.

        Ensure that your response is strictly based on the provided context and does not include any information outside of it. If you have already answered a question, do not repeat the same answer unless a new question is asked. If the question is not related to the context, politely inform the user that you cannot provide an answer.
        """,
        'chroma_path': "./db/chroma/osa"
    }
}

CHROMA_PATH = "./db/chroma/ED"

class Memory:
    def __init__(self):
        self.history = []

    def add_message(self, message):
        self.history.append(message)

    def get_history(self):
        return "\n".join(self.history)

app = Flask(__name__)
memory = Memory()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/reset', methods=['POST'])
def reset_memory():
    global memory
    memory = Memory()  # Reset the memory instance
    print("`Memory` has been reset successfully.")
    return jsonify({'status': 'Memory reset successfully'})

@app.route('/ask', methods=['POST'])
def ask():
    query_text = request.json.get('query')
    chatbot_id = request.json.get('chatbot_id', 'chatbot_1')  # Default to chatbot_1 if not specified
    response_text, sources = query_rag(query_text, chatbot_id)
    return jsonify({'response': response_text, 'sources': sources})


def query_rag(query_text: str, chatbot_id: str):
    # Get the configuration for the selected chatbot
    config = CHATBOT_CONFIGS.get(chatbot_id)

    if not config:
        return "Invalid chatbot selection.", []

    prompt_template = config['prompt']
    chroma_path = config['chroma_path']

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    # Limit the number of documents retrieved
    results = db.similarity_search_with_score(query_text, k=5)  # Get top 5 results
    if not results:
        return "I apologize, but there is no question provided. Please provide a question for me to answer based on the given context.", []

    # Gather context from results
    context_texts = [doc.page_content for doc, _score in results]

    # Combine context texts and limit size
    context_text = "\n\n---\n\n".join(context_texts)

    # Limit the context size to a certain number of characters (e.g., 1000 characters)
    max_context_length = 1000
    if len(context_text) > max_context_length:
        context_text = context_text[:max_context_length] + "..."  # Truncate and add ellipsis

    # Construct the prompt
    prompt_template = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Invoke the model
    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    # Update memory with the latest user and assistant messages
    memory.add_message(f"User: {query_text}")
    memory.add_message(f"Assistant: {response_text}")

    # Retrieve sources from the results
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"{response_text}"

    return formatted_response, sources

if __name__ == "__main__":
    app.run(debug=True)
