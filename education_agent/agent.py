import os
import asyncio
import google.auth
from google.cloud import bigquery, storage
from google import genai
from google.genai import types
from google.adk.tools.google_search_tool import google_search
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.bigquery import BigQueryCredentialsConfig, BigQueryToolset
from google.adk.tools.bigquery.config import BigQueryToolConfig, WriteMode
from uuid import uuid4
import vertexai

# --- Constants ---
# Please replace with your actual Project ID
PROJECT_ID = "qwiklabs-gcp-03-f038a3e064bd" 
LOCATION = "us-central1"
DATASET_NAME = "education"

APP_NAME = "education-orchestrator-app"

# --- One-time Setup: Bucket, Vertex AI, BigQuery Auth ---
# This section will run only once when the Streamlit app starts.
STAGING_BUCKET = None
try:
    print("Initializing GCP services...")
    storage_client = storage.Client(project=PROJECT_ID)
    bucket_name = f"agent-engine-staging-{uuid4().hex[:12]}"
    bucket = storage_client.create_bucket(bucket_name, location="US")
    STAGING_BUCKET = bucket.name
    print(f"Successfully created and using staging bucket: gs://{STAGING_BUCKET}")

    vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=f"gs://{STAGING_BUCKET}")
    print("Vertex AI SDK initialized.")

    # --- FIX: Set environment variables to configure genai for Vertex AI ---
    # This is the backward-compatible way to ensure the ADK uses the Vertex AI backend.
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
    os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION
    print("Set environment variables for GenAI to use Vertex AI.")

    creds, proj = google.auth.default()
    bq_client = bigquery.Client(project=PROJECT_ID, credentials=creds)
    bq_client.query("SELECT 1").result()
    print("BigQuery authentication and connection successful.")
    print("Initialization complete.")

except Exception as e:
    print(f"FATAL: An error occurred during setup: {e}")
    print("Please ensure your GCP project is correctly configured and you have authenticated.")
    # In a web app, you might want to show an error state on the UI
    # For simplicity, we'll exit if setup fails.
    exit()

MODEL = "gemini-2.5-flash"

# --- Tool Configuration ---
adc, _ = google.auth.default()
bq_credentials = BigQueryCredentialsConfig(credentials=adc)
bq_tool_cfg = BigQueryToolConfig(write_mode=WriteMode.BLOCKED)
bq_tools = BigQueryToolset(credentials_config=bq_credentials, bigquery_tool_config=bq_tool_cfg)


# --- AGENT 1: The Database Specialist ---
DATA_AGENT_INSTR = f"""
You are a highly specialized data analyst agent. Your ONLY capability is to query a BigQuery database.
- You must always first inspect the schema of the dataset `{PROJECT_ID}.{DATASET_NAME}` to see available tables.
- Formulate and execute a `SELECT` SQL query based on the user's question.
- Fully qualify every table name as `{PROJECT_ID}.{DATASET_NAME}.<table>`.
- Never perform DDL/DML operations (e.g., CREATE, ALTER, INSERT, DELETE).
- If your query returns a valid result, provide the answer and the SQL you used.
- **CRITICAL**: If you determine that the database cannot answer the question (either because no relevant tables exist or the query returns no results), you MUST reply with ONLY the following exact phrase and nothing else: `NO_DATA_FOUND`
"""

data_agent = Agent(
    model=MODEL,
    name="DataAgent",
    description="Exclusively queries a BigQuery dataset to find information.",
    instruction=DATA_AGENT_INSTR,
    tools=[bq_tools]
)


# --- AGENT 2: The Web Search Specialist ---
SEARCH_AGENT_INSTR = """
You are a highly specialized web search assistant. Your ONLY capability is to use Google Search to answer questions.
- You cannot access any databases.
- Provide a concise and helpful answer to the user's question based on the search results.
"""

search_agent = Agent(
    model=MODEL,
    name="SearchAgent",
    description="Exclusively uses Google Search to find information from the web.",
    instruction=SEARCH_AGENT_INSTR,
    tools=[google_search]
)

# --- Session Management and Runners for each agent ---
session_service = InMemorySessionService()

data_runner = Runner(agent=data_agent, app_name=APP_NAME, session_service=session_service)
search_runner = Runner(agent=search_agent, app_name=APP_NAME, session_service=session_service)


# --- Generic function to call an agent ---
async def call_agent_async(query: str, runner, user_id, session_id):
    """Sends a query to a specified agent runner and returns the final response."""
    content = types.Content(role='user', parts=[types.Part(text=query)])
    final_response_text = "Agent did not produce a final response."

    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            break
    return final_response_text


# --- THE ORCHESTRATOR ---
async def orchestrator(query: str, user_id: str, session_id: str):
    """
    Manages the workflow by calling agents in a sequence.
    1. First, it asks the DataAgent.
    2. If the DataAgent returns a specific failure signal, it asks the SearchAgent.
    """
    print(f"[{session_id}] - User Query: \"{query}\"")
    
    print(f"[{session_id}] - Step 1: Consulting the DataAgent...")
    data_response = await call_agent_async(query, data_runner, user_id, session_id)
    
    if "NO_DATA_FOUND" in data_response:
        print(f"[{session_id}] - Signal 'NO_DATA_FOUND' received. Falling back to SearchAgent.")
        print(f"[{session_id}] - Step 2: Consulting the SearchAgent...")
        search_response = await call_agent_async(query, search_runner, user_id, session_id)
        return search_response
    else:
        print(f"[{session_id}] - Success! DataAgent found a conclusive answer.")
        return data_response


# --- Web App Entry Point ---
async def call_agent(query: str) -> str:
    """
    This is the main entry point for the Streamlit web application.
    It creates a unique session for each call and runs the orchestrator.
    """
    # For a production app, you might want to manage user and session IDs
    # more robustly, perhaps using Streamlit's session_state.
    user_id = "webapp-user"
    session_id = str(uuid4())
    
    # The session must be created before the runner can use it.
    await session_service.create_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    
    return await orchestrator(query, user_id, session_id)

