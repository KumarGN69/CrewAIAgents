from crewai import Crew, Process
from crewai.memory import EntityMemory
from crewai import Agent
from tasks import research_task, writer_task
from agents import blog_reseacher, blog_writer

## create the crew and link the agents and tasks
crew = Crew(
    tasks = [research_task,writer_task],
    agents = [blog_reseacher, blog_writer],
    process = Process.sequential,
    memory = True,
    cache = True,
    max_rpm= 100,
    share_crew= True,
    memory_args={
        "short_term": None
        }
    
    )


##start the execution procees with enhanced feedback configuration
result = crew.kickoff(inputs={'topic':'Top stories'})
print(result)