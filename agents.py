from crewai import Agent, LLM
from tools import yt_tool, scrape_tool
import os
from dotenv import load_dotenv

load_dotenv()

print(os.getenv('OPENAI_MODEL_NAME'))
print(os.getenv('OPENAI_API_BASE'))
print(os.getenv('OPENAI_API_KEY'))

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
os.environ["OPENAI_MODEL_NAME"] = os.getenv("OPENAI_MODEL_NAME")

# print(os.environ['OPENAI_MODEL_NAME'])
# print(os.environ['OPENAI_API_BASE'])
# print(os.environ['OPENAI_API_KEY'])


##Create a senior blog reseacher

blog_reseacher = Agent(
    role='Blog Researcher from Google news website ',
    goal='get the relevant content for the topic {topic} from Google News website ',
    backstory="""You are an expert in understanding the Politics and happenings in the world """,
    verbose=True,
    memory=True,
    tools=[scrape_tool],
    allow_delegation=True,
    llm=LLM(model="ollama/llama3.2", base_url="http://localhost:11434")
)

##Create a senior blog content write

blog_writer = Agent(
    role='Blog writer',
    goal='Narrate compelling news stories about the topic {topic} from Google news website ',
    backstory="""with a flair for simplifying complex topics, you craft engaging narratives that captivate, educate,
                 and bring new discoveries to light in an accessible manner""",
    verbose=True,
    memory=True,
    tools=[scrape_tool],
    allow_delegation=False,
    llm=LLM(model="ollama/llama3.2", base_url="http://localhost:11434")
)
