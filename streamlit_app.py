# ADHD Coach Chatbot with Todoist Integration - Streamlit App
# This file integrates Todoist for task management and uses OpenAI to offer ADHD coaching.

# --- 1. Imports and Setup ---
import os
import requests
from requests_oauthlib import OAuth2Session
from http.server import BaseHTTPRequestHandler, HTTPServer
import webbrowser
import openai
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from todoist_api_python.api import TodoistAPI
import PyPDF2
from dotenv import load_dotenv
from openai import OpenAIError

# Load environment variables from the .env file
load_dotenv()
client_id = os.getenv("TODOIST_CLIENT_ID")
client_secret = os.getenv("TODOIST_CLIENT_SECRET")
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

# Configuration details
redirect_uri = "http://localhost:8010/callback"  # Ensure this matches your Todoist redirect URI
authorization_base_url = "https://todoist.com/oauth/authorize"
token_url = "https://todoist.com/oauth/access_token"
scope = "data:read_write"

# Directory containing ADHD textbooks in PDF format
pdf_directory = "/Users/kripanavas/Downloads/ADHD-Coach-main/rag"

# --- Define the Task and Due classes globally ---
class Task:
    def __init__(self, content, due):
        self.content = content
        self.due = due

class Due:
    def __init__(self, date):
        self.date = date

# --- 2. Todoist OAuth Flow ---
# Function to fetch access token after authorization
def fetch_access_token(auth_code, retries=3, delay=2):
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": auth_code,
        "redirect_uri": redirect_uri,
    }
    for attempt in range(retries):
        print(f"Attempt {attempt + 1} to fetch access token...")  # Debugging statement
        try:
            response = requests.post(token_url, data=data, timeout=5)
            print("Access token response:", response.status_code)  # Debugging statement
            if response.status_code == 200:
                token = response.json()
                print("Access token retrieved:", token['access_token'])  # Debugging statement
                return token['access_token']
            else:
                print("Failed to retrieve access token:", response.status_code, response.text)
                return None
        except requests.exceptions.Timeout:
            print(f"Attempt {attempt + 1} timed out. Retrying in {delay} seconds...")
            time.sleep(delay)
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None
    print("All attempts to fetch access token have failed. Calling backup method.")
    return backup_access_token_method()

def backup_access_token_method():
    print("Executing backup method for access token.")
    # Since access token is essential, you might need to re-initiate the OAuth flow
    # or inform the user to try again later
    # For now, we'll return None to indicate failure
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
            print("Authorization code received:", authorization_code)  # Debugging statement
            access_token = fetch_access_token(authorization_code)
            self.server.token = {"access_token": access_token} if access_token else None

# Start server to handle OAuth callback
server_address = ('', 8010)
httpd = HTTPServer(server_address, OAuthCallbackHandler)
print("Waiting for the callback with the authorization code...")
httpd.handle_request()
print("Callback received.")  # Debugging statement

# Retrieve and set access token
if hasattr(httpd, 'token') and httpd.token:
    access_token = httpd.token['access_token']
    headers = {'Authorization': f"Bearer {access_token}"}
    print("Access token retrieved.")
else:
    print("Failed to retrieve access token.")
    access_token = None  # Ensure access_token is defined

# --- 3. Todoist Task Retrieval ---
def fetch_tasks():
    """
    Attempts to fetch tasks from the Todoist API with a timeout.
    If the call fails or times out, it uses the backup method.
    """
    if not access_token:
        print("No access token available. Using backup method.")
        return fetchMocked()
    try:
        url = 'https://api.todoist.com/rest/v2/tasks'
        headers = {'Authorization': f'Bearer {access_token}'}
        # Set timeout to 5 seconds
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()  # Raises an HTTPError if the response was unsuccessful
        data = response.json()
        tasks = []
        for item in data:
            content = item.get('content')
            due_data = item.get('due')
            due_date = due_data.get('date') if due_data else None
            due = Due(due_date) if due_date else None
            tasks.append(Task(content, due))
        if tasks:
            for task in tasks:
                print(f"Task: {task.content} - Due: {task.due.date if task.due else 'No due date'}")
        else:
            print("No tasks found.")
        return tasks
    except requests.exceptions.Timeout:
        print("API call timed out. Using backup method.")
        return fetchMocked()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return fetchMocked()

def fetchMocked():
    """
    Provides a backup list of tasks in case the API call fails.
    """
    print("Executing backup method.")
    # Create a list of tasks with and without due dates
    tasks = [
        Task('Finish the quarterly report', Due('2023-10-15')),
        Task('Email the client about the new proposal', None),
        Task('Prepare slides for the presentation', Due('2023-10-20')),
        Task('Organize team-building activity', None),
    ]
    return tasks

# --- Initialize tasks in session state ---
if 'tasks' not in st.session_state:
    st.session_state.tasks = []  # Initialize as an empty list in session state

# --- Fetch tasks if not already done ---
if not st.session_state.tasks:  # Only fetch if `tasks` is empty
    st.session_state.tasks = fetch_tasks()  # Populate tasks in session state

# Set a sample task for testing if no tasks are returned
sample_task = st.session_state.tasks[0] if st.session_state.tasks else None
user_input = "I'm struggling to focus on my task."


# --- 4. Chatbot Response Generation ---
def generate_response(task, user_input):
    task_title = task.content if hasattr(task, 'content') else "Sample Task"
    task_description = getattr(task, 'description', "No description available")
    prompt = f"You have a task: {task_title}. Details: {task_description}. {user_input}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
    except OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return "I'm sorry, I encountered an issue generating a response."
    
# Generate a response based on the availability of a sample task
if sample_task:
    response = generate_response(sample_task, user_input)
else:
    # Fallback for when no tasks are available
    mock_task = Task("Graduation PPT", None)
    response = generate_response(mock_task, user_input)

print("Kira's response:", response)


# --- 5. RAG Setup with ADHD Textbooks ---
def extract_text_from_pdf(pdf_file):
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages])
    return text

def load_and_preprocess_texts():
    texts = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_directory, filename)
            texts.append(extract_text_from_pdf(file_path))
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

# Retrieve relevant docs for user query
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
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
    except OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return "I'm sorry, I encountered an issue generating a response."


# --- 7. Streamlit Interface ---
st.title("Kira: Your ADHD Coach Chatbot")
st.write("Ask Kira for support with your tasks and ADHD-related guidance.")

# Input fields for user to enter task and query
task_title = st.text_input("What is your task?", placeholder="e.g., 'Complete my graduation PPT'")
user_input = st.text_input("What would you like help with?", placeholder="I need help focusing on this task")

# Display Kira's response
if st.button("Ask Kira"):
    if task_title and user_input:
        response = generate_coach_response_with_rag(task_title, user_input)
        st.write(f"**Kira**: {response}")
    else:
        st.write("Please enter both a task and your question for Kira.")
