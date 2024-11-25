import os
from dotenv import load_dotenv

from crewai import Agent , Task, Process, Crew, LLM
from crewai_tools import ScrapeWebsiteTool


load_dotenv()

print(os.getenv('OPENAI_MODEL_NAME'))
print(os.getenv('OPENAI_API_BASE'))
print(os.getenv('OPENAI_API_KEY'))

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
os.environ["OPENAI_MODEL_NAME"] = os.getenv("OPENAI_MODEL_NAME")

scrape_tool = ScrapeWebsiteTool(website_url="https://docs.crewai.com/how-to/Installing-CrewAI/",
    config=dict(
        llm=dict(
            provider="ollama",
            config=dict(
                model="llama3.2"  # Ensure this matches exactly
            ),
        ),
        embedder=dict(
            provider="ollama",  # or another provider if applicable
            config=dict(
                model="llama3.2"  # Ensure this matches exactly
            ),
        ),
    )
)

blog_reseacher = Agent(
    role='Blog Researcher by searching data from a website ',
    goal='get the relevant content for the topic {topic} from a website ',
    backstory="""You are an expert in understanding the technology and software """,
    verbose=True,
    memory=True,
    tools=[scrape_tool],
    allow_delegation=True,
    llm=LLM(model="ollama/llama3.2", base_url="http://localhost:11434"),
    max_iter =2
)



research_task = Task(
    description = "Identify details on the topic {topic}, Get detailed information from the website",
    expected_output= "A bulleted list long report based on the {topic} and create the content for the summary",
    tools=[scrape_tool],
    agent= blog_reseacher
)

crew = Crew(
    tasks = [research_task],
    agents = [blog_reseacher],
    process = Process.sequential,
    memory = True,
    cache = True,
    max_rpm= 1,
    share_crew= True,
    memory_args={
        "short_term": None
        }
    )


##start the execution procees with enhanced feedback configuration
result = crew.kickoff(inputs={'topic':'How to implement Custom Tools'})

print(result)
