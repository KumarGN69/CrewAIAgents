from crewai import Task
from tools import yt_tool, scrape_tool
from agents import blog_reseacher, blog_writer

#Create a researcher task
research_task = Task(
                    description = "Identify news on the topic{topic},Get detailed information from the website",
                    expected_output= "A comprehensive 3 paragraphs of long report based on the {topic} and create the content for the summary",
                    tools=[scrape_tool],
                    agent= blog_reseacher
)

#Create a writer task
writer_task = Task(
                    description = " get the info from the Google news website on the topic{topic}",
                    expected_output= "Summarize the info into three paragraphs from Google News website topic{topic} ",
                    tools=[scrape_tool],
                    agent= blog_writer,
                    async_execution= False,
                    output_file="new_blog_post.md"
)

