from crewai import Crew, Agent, Task, Process,LLM
# from langchain_community.llms import ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
# from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import tool
from crewai_tools import Tool

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
os.environ["OPENAI_MODEL_NAME"] = os.getenv("OPENAI_MODEL_NAME")

# Website Data Ingestion 
loader = UnstructuredURLLoader(urls=["https://docs.crewai.com/how-to/Installing-CrewAI/"])
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(data)
# print(texts)

#initialize embeddings
MODEL_NAME = "nomic-embed-text"
embeddings = OllamaEmbeddings(model= MODEL_NAME)

# Create and Persist Vector Database
db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
db.persist()

# Define Retriever from Vector Store
retriever = db.as_retriever()

@tool
def retrieve_docs():
    """Useful to retrive docs from vectordb together."""
    return retriever.get_relevant_documents()

#define agents and tasks

researcher = Agent(
    # ai=ollama_wrapper,
    name="Researcher",
    role="Researches topics by searching the website data.",
    tools=[ retrieve_docs
        # Tool(
        #     name="Website_Search",
        #     func=retriever.get_relevant_documents,
        #     description="useful for when you need to ask with lookup on website data."
        # )
    ],
    llm=LLM(model="ollama/llama3.2", base_url="http://localhost:11434"),
    goal="Answer questions by retrieving relevant information from the website's data.",  # Add the goal here
    backstory="You are a helpful AI assistant specializing in searching and retrieving information from a website. Use your 'Website_Search' tool to find relevant documents when answering questions."  
)

researcher_boss = Agent(
    # ai=ollama_wrapper,
    name="Researcher Boss",
    role="Challenges the researcher to bring out the best out of his findings",
    tools=[ retrieve_docs
        # Tool(
        #     name="Website_Search",
        #     func=retriever.get_relevant_documents,
        #     description="useful for when you need to ask with lookup on website data."
        # )
    ],
    llm=LLM(model="ollama/llama3.2", base_url="http://localhost:11434"),
    goal="Ask further questions to the researcher and validates the retrieved relevant information from the website's data.",  # Add the goal here
    backstory="You are a helpful AI assistant boss and your job is to make sure the retrieved information is correct. Use your 'Website_Search' tool to find relevant documents when answering questions."  
)


research_task = Task(
    description=(
        "Analyze the URL provided ({crewai_url}) "
        "to extract information about how crewai works. "
        "required. Use the tools to gather content and identify "
        "and categorize the requirements."
    ),
    expected_output=(
        "A structured list of crewai specifications, including necessary "
        "tools to get started"
    ),
    agent=researcher,
    async_execution=True
    
)

boss_task = Task(
    description=(
        "Analyze the URL provided ({crewai_url}) "
        "to extract information about how crewai works. "
        "required. Use the tools to gather content and identify "
        "and categorize the requirements."
    ),
    expected_output=(
        "A structured list of crewai specifications, including necessary "
        "tools to get started"
    ),
    agent=researcher_boss,
    async_execution=True
)

# Create Crew
research_crew1 = Crew(
    agents=[researcher, researcher_boss],
    tasks=[research_task, boss_task],
    verbose=True,  # This will print logs to the console as the crew works
    process = Process.sequential,
    memory = True,
    cache = True,
    max_rpm= 100,
    share_crew= True,
    memory_args={
        "short_term": None
        }
)

# Job Context
job_crew_works = {
    'crewai_url': 'https://docs.crewai.com/how-to/Installing-CrewAI/',
    'personal_writeup': """Accomplished Researcher 
    with 18 years of experience, specializing in
    setting up CrewAI kind of agent based systems"""
}

# Kickoff the Crew's Work
result = research_crew1.kickoff(inputs=job_crew_works)
print(result)