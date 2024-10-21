# ADHD Coach Chatbot with Todoist Integration - Streamlit App
# This file integrates Todoist for task management and uses OpenAI to offer ADHD coaching.

# --- 1. Imports and Setup ---
import os
import requests
from requests_oauthlib import OAuth2Session
from http.server import BaseHTTPRequestHandler, HTTPServer
import webbrowser
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from todoist_api_python.api import TodoistAPI
import PyPDF2
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
import os
import json
import openai

# Load environment variables from the .env file
load_dotenv()
client_id = os.getenv("TODOIST_CLIENT_ID")
client_secret = os.getenv("TODOIST_CLIENT_SECRET")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of the OpenAI client
client = OpenAI(api_key=openai_api_key)

# Configuration details
redirect_uri = "http://localhost:8010/callback"  # Ensure this matches your Todoist redirect URI
authorization_base_url = "https://todoist.com/oauth/authorize"
token_url = "https://todoist.com/oauth/access_token"
scope = "data:read_write"

# Directory containing ADHD textbooks in PDF format
pdf_directory = "/Users/kripanavas/Downloads/ADHD-Coach-main/rag"

# Define the path for storing the token
token_file_path = 'todoist_token.json'

# --- Define the Task and Due classes globally ---
class Task:
    def __init__(self, content, due):
        self.content = content
        self.due = due

class Due:
    def __init__(self, date):
        self.date = date

# Step 1: Load or Retrieve OAuth Token
def load_token():
    # Load token from the file if it exists
    if os.path.exists(token_file_path):
        with open(token_file_path, 'r') as token_file:
            token_data = json.load(token_file)
            return token_data.get("access_token")
    return None

def save_token(access_token):
    # Save token to a file
    with open(token_file_path, 'w') as token_file:
        json.dump({"access_token": access_token}, token_file)

# Streamlit session state for access token
if 'access_token' not in st.session_state:
    st.session_state.access_token = load_token()

# Step 2: Authorization - Only required if no token
if not st.session_state.access_token:
    # Function to fetch access token after authorization
    def fetch_access_token(auth_code, retries=3, delay=2):
        data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": auth_code,
            "redirect_uri": redirect_uri,
        }
        for attempt in range(retries):
            try:
                response = requests.post(token_url, data=data, timeout=5)
                if response.status_code == 200:
                    token = response.json()
                    return token['access_token']
                else:
                    print("Failed to retrieve access token:", response.status_code, response.text)
                    return None
            except requests.exceptions.Timeout:
                time.sleep(delay)
            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}")
                return None
        return None

    # Start OAuth2 session
    todoist = OAuth2Session(client_id, redirect_uri=redirect_uri, scope=[scope])
    authorization_url, state = todoist.authorization_url(authorization_base_url)
    print("Go to the following URL to authorize:", authorization_url)
    webbrowser.open(authorization_url)

    # Define callback handler for local server
    class OAuthCallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"You can close this window.")
            query = self.path.split('?', 1)[-1]
            params = dict(qc.split('=') for qc in query.split('&'))
            authorization_code = params.get("code")
            if authorization_code:
                access_token = fetch_access_token(authorization_code)
                if access_token:
                    st.session_state.access_token = access_token
                    save_token(access_token)

    # Start server to handle OAuth callback
    server_address = ('', 8010)
    httpd = HTTPServer(server_address, OAuthCallbackHandler)
    print("Waiting for the callback with the authorization code...")
    httpd.handle_request()
    print("Callback received.")  # Debugging statement

# Retrieve and set access token in headers if available
if st.session_state.access_token:
    access_token = st.session_state.access_token
    headers = {'Authorization': f"Bearer {access_token}"}
else:
    st.error("Failed to retrieve access token. Please restart the application and try authorizing again.")

# --- 3. Todoist Task Retrieval ---
def fetch_tasks():
    if not access_token:
        print("No access token available. Using backup method.")
        return fetchMocked()
    try:
        url = 'https://api.todoist.com/rest/v2/tasks'
        headers = {'Authorization': f'Bearer {access_token}'}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        tasks = [Task(item.get('content'), Due(item.get('due').get('date')) if item.get('due') else None) for item in data]
        return tasks
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return fetchMocked()

def fetchMocked():
    tasks = [
        Task('Finish the quarterly report', Due('2023-10-15')),
        Task('Email the client about the new proposal', None),
        Task('Prepare slides for the presentation', Due('2023-10-20')),
        Task('Organize team-building activity', None),
    ]
    return tasks

# --- Initialize tasks in session state ---
if 'tasks' not in st.session_state:
    st.session_state.tasks = fetch_tasks()

# Set a sample task for testing if no tasks are returned
sample_task = st.session_state.tasks[0] if st.session_state.tasks else None
user_input = "I'm struggling to focus on my task."


# --- 4. Chatbot Response Generation ---
def generate_response(task, user_input):
    task_title = task.content if hasattr(task, 'content') else "Sample Task"
    task_description = getattr(task, 'description', "No description available")
    prompt = f"You have a task: {task_title}. Details: {task_description}. {user_input}"
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return "I'm sorry, I encountered an issue generating a response."

# Generate a response based on the availability of a sample task
if sample_task:
    response = generate_response(sample_task, user_input)
else:
    mock_task = Task("Graduation PPT", None)
    response = generate_response(mock_task, user_input)

print("Kira's response:", response)

# Load OpenAI API key from the environment or set it directly
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

# --- 5. RAG Setup with ADHD Textbooks ---
def extract_text_from_pdf(pdf_file):
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages])
    return text

def load_and_preprocess_texts():
    texts = [extract_text_from_pdf(os.path.join(pdf_directory, filename)) for filename in os.listdir(pdf_directory) if filename.endswith(".pdf")]
    return "\n\n".join(texts)

def setup_vectorstore():
    try:
        full_text = load_and_preprocess_texts()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.create_documents([full_text])
        embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
        return FAISS.from_documents(documents, embeddings)
    except Exception as e:
        print(f"Error setting up vector store: {e}")
        return None

vectorstore = setup_vectorstore()

def retrieve_relevant_docs(query):
    try:
        return vectorstore.similarity_search(query, k=5)
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []

# --- 6. Generate Response with RAG ---
def generate_coach_response_with_rag(task_title, user_input):
    relevant_docs = retrieve_relevant_docs(user_input)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"Use the following ADHD textbook information:\n\n{context}\n\n" \
             f"The user has the following task: {task_title}. The user says: {user_input}"
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return "I'm sorry, I encountered an issue generating a response."


# --- 7. Streamlit Interface ---
st.title("Kira: Your ADHD Coach Chatbot")
st.write("Ask Kira for support with your tasks and ADHD-related guidance.")

# Input fields for user to enter task and query
task_title = st.text_input("What is your task?", placeholder="e.g., 'Complete my graduation PPT'")
user_input = st.text_input("What would you like help with?", placeholder="I need help focusing on this task")

# Create a placeholder for the response that can be updated later
response_placeholder = st.empty()

# Check if 'kira_response' exists in the session state, if yes, display it
if "kira_response" in st.session_state and st.session_state.kira_response:
    response_placeholder.write(f"**Kira**: {st.session_state.kira_response}")

# Button to trigger Kira's response
if st.button("Ask Kira"):
    # Check if the user has provided both a task and a question
    if not task_title or not user_input:
        st.error("Please enter both a task and your question for Kira.")
    elif not st.session_state.get("access_token"):
        st.error("Please complete the authorization process first.")
    else:
        # Generate response using the RAG-based model
        try:
            response = generate_coach_response_with_rag(task_title, user_input)
            if response:
                st.markdown(f"**Kira**: {response}")  # Display Kira's response directly under the button
            else:
                st.write("Kira could not generate a response. Please try again.")
        except Exception as e:
            st.write(f"An error occurred: {e}")
            # Display Kira's response if it exists in session state
if 'kira_response' in st.session_state:
    st.text_area("Kira's Response", st.session_state['kira_response'], height=150)
