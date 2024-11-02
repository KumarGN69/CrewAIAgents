from crewai_tools import YoutubeChannelSearchTool,ScrapeWebsiteTool

# yt_tool = YoutubeChannelSearchTool(youtube_channel_handle="@krishnaik06")
yt_tool = YoutubeChannelSearchTool(youtube_channel_handle="@krishnaik06",
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

scrape_tool = tool = ScrapeWebsiteTool(website_url='https://news.google.com/home?hl=en-US&gl=US&ceid=US:en',
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

# yt_tool = YoutubeChannelSearchTool(youtube_channel_handle='@krishnaik06',
#     config=dict(
#         llm=dict(
#             provider="ollama", # or google, openai, anthropic, llama2, ...
#             config=dict(
#                 model="ollama/llama3.2",
#                 # temperature=0.5,
#                 # top_p=1,
#                 # stream=true,
#             ),
#         ),
#         embedder=dict(
#             provider="ollama", # or openai, ollama, ...
#             config=dict(
#                 model="models/embedding-001",
#                 # task_type="retrieval_document",
#                 # title="Embeddings",
#             ),
#         ),
#     )
# )
