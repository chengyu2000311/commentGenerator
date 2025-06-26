import base64
import os
import argparse
import uuid

from langchain.chat_models import init_chat_model
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import Tool
from langchain.prompts import PromptTemplate

from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import (
    create_sync_playwright_browser,
)
from langchain_community.tools.reddit_search.tool import RedditSearchRun
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper


def configure_environment():
    """
    Load required following API keys and environment variables.

    GOOGLE_API_KEY=
    REDDIT_CLIENT_ID=
    REDDIT_CLIENT_SECRET=
    REDDIT_USER_AGENT=
    """
    secrets_path = os.path.join(os.path.dirname(__file__), "secrets.txt")
    if not os.path.exists(secrets_path):
        raise FileNotFoundError(f"secrets.txt not found at {secrets_path}")
    with open(secrets_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition("=")
            if sep != "=":
                continue
            os.environ[key] = value.strip()


class NewsCommentAgent:
    def __init__(
        self,
        llm_model: str = "gemini-2.0-flash",
        llm_image_model: str = "gemini-2.0-flash-preview-image-generation",
        temperature: float = 0.7
    ):
        self.llm = init_chat_model(
            llm_model,
            model_provider="google_genai",
            temperature=temperature
        )
        self.llm_image = init_chat_model(
            llm_image_model,
            model_provider="google_genai",
            temperature=temperature
        )

        self.browser = create_sync_playwright_browser()
        self.reddit_tool = self._init_reddit_tool()
        self.tools = self._init_tools()
        self.prompt = self._build_prompt()
        self.executor = self._create_executor()

    def _init_reddit_tool(self) -> RedditSearchRun:
        return RedditSearchRun(
            api_wrapper=RedditSearchAPIWrapper(
                reddit_client_id=os.environ["REDDIT_CLIENT_ID"],
                reddit_client_secret=os.environ["REDDIT_CLIENT_SECRET"],
                reddit_user_agent=os.environ["REDDIT_USER_AGENT"],
            )
        )

    def simple_reddit_search(self, query: str) -> str:
        """
        Search Reddit for example comments based on a query.
        """
        params = {
            "query": query,
            "subreddit": "all",
            "sort": "relevance",
            "time_filter": "week",
            "limit": '5'
        }
        return self.reddit_tool.run(params)

    def _init_tools(self) -> list:
        toolkit = PlayWrightBrowserToolkit.from_browser(
            sync_browser=self.browser
        )
        tools = toolkit.get_tools()
        wrapped_tools = []
        for t in tools:
            if t.name == "extract_hyperlinks":
                original = t
                def _extract_links(url: str, _orig=original):
                    return _orig.run({
                        "url": url,
                        "absolute_urls": False
                    })
                wrapped_tools.append(
                    Tool(
                        name="extract_hyperlinks",
                        func=_extract_links,
                        description=original.description
                    )
                )
            else:
                wrapped_tools.append(t)
        
        wrapped_tools.append(
            Tool(
                name="reddit_search",
                func=self.simple_reddit_search,
                description=(
                    "Search Reddit comments for a given topic; "
                    "defaults: subreddit=all, sort by relevance, past week, limit=5"
                )
            )
        )

        wrapped_tools.append(
            Tool(
                name="generate_image",
                func=self._generate_image,
                description=(
                    "Call this to generate an image from text: "
                    "input is a prompt string, output is relative file path of PNG image."
                )
            )
        )
        return wrapped_tools

    def _build_prompt(self) -> PromptTemplate:
        template = '''Answer the following questions as best you can. You have access to the following tools:

    {tools}

    You are an experienced netizens that like to leave highly engaging and insightful comments (“神评论”) on specified news articles. Your responsibilities include:

    1. **Understand the Post**  
    - Visit the given news url.
    - Comprehend both the textual content and any images in the article.  
    - Identify humor points, criticisms, core viewpoints, and potential controversies.

    2. **Retrieve Inspirational Comments**  
    - Search Reddit for the most relevant, high-engagement comments on the same topic to learn patterns.

    3. **Generate Original Engaging Comments**  
    - Produce comments that stimulate user interaction, with varied styles (provocative, concise & sharp, humorous, thought-provoking) and within 30 words in **English**.
        - Provocative
        - Concise & Sharp
        - Witty Joke & Sarcastic
        - Thought-Provoking Question

    4. **Image Description Generation/Matching**  
    - Propose or generate an image concept to enhance the comment's impact using a sentence.

    5. **Generate Images**  
    - For each comment's “image_idea”, call the tool:  
        ```
        Action: _generate_image
        Action Input: <your image_idea string>
        Observation: <relative file path to PNG Image>
        ```  
    - Then include those file paths in each comment in your Final Answer.

    **Output Requirement:**
    - Your **Final Answer** must be a JSON array of exactly four objects, one for each style above, in this order.
    - Each object should have the following fields:
        - **"style"**: one of the four Chinese labels.
        - **"comment"**: the generated comment text.
        - **"image_idea"**: a brief description of a matching image.
        - **"image_file_path"**: the relative file path to the PNG image.

    Use the following format:

    Question: the input question you must answer  
    Thought: you should always think about what to do  
    Action: the action to take, should be one of [{tool_names}]  
    Action Input: the input to the action  
    Observation: the result of the action  
    … (this Thought/Action/Action Input/Observation sequence can repeat N times)  
    Thought: I now know the final answer  
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}  
    Thought:{agent_scratchpad}'''

        return PromptTemplate.from_template(template)

    def _create_executor(self) -> AgentExecutor:
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    

    def _generate_image(self, description: str) -> str:
        """
        Generate an image from text using the ChatGoogleGenerativeAI image model.
        Returns file path to the PNG image.
        """
        message = {"role": "user", "content": description}
        response: AIMessage = self.llm_image.invoke(
            [message],
            generation_config={"response_modalities": ["TEXT", "IMAGE"]}
        )
        # Extract base64 from the first image_url block
        uri = next(
            block["image_url"]["url"]
            for block in response.content
            if isinstance(block, dict) and block.get("image_url")
        )
        # write the raw image off to disk
        _, b64 = uri.split(",", 1)
        img_bytes = base64.b64decode(b64)
        path = f"./{uuid.uuid4().hex}.png"
        with open(path, "wb") as f:
            f.write(img_bytes)
        return path

    def run(self, news_url: str):
        """
        Execute the agent on the given news URL and return the results.
        """
        print(f"Processing: {news_url}\n")
        try:
            return self.executor.invoke({"input": news_url})
        finally:
            self.browser.close()


def main():
    configure_environment()

    parser = argparse.ArgumentParser(
        description="Generate engaging comments for a news article."
    )
    parser.add_argument(
        "url",
        help="The news article URL to process.",
        default="https://www.ainvest.com/news/nato-commits-5-gdp-defence-president-trump-backs-bigger-military-budgets-russian-threat-2506/"
    )
    args = parser.parse_args()

    agent = NewsCommentAgent()
    response = agent.run(args.url)


    out_path = "comments_output.json"
    with open(out_path, "w") as f:
        f.write(response['output'])

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
