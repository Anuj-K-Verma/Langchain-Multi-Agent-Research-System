from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()

# Search agent: moved to gpt-oss-20b — qwen's 7000 ITPM limit was too tight
search_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=900
)

# Reader agent: needs to read/summarize scraped content — give it more room
reader_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=2000
)

# Writer chain: needs to write a FULL detailed report — needs the most tokens
writer_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3,
    max_tokens=3000
)

# Critic chain: short structured feedback — small output is fine, qwen works here
critic_llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    max_tokens=700
)

# System guard to stop gpt-oss models from hallucinating built-in browser tools
TOOL_GUARD = (
    "You only have access to the tools explicitly given to you. "
    "Never call any other tool, including any browser, search, or open tool "
    "that is not in your provided tool list."
)


def build_search_agent():
    return create_agent(
        model=search_llm,
        tools=[web_search],
        system_prompt=TOOL_GUARD,
    )

def build_reader_agent():
    return create_agent(
        model=reader_llm,
        tools=[scrape_url],
        system_prompt=TOOL_GUARD,
    )

# writer chain
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | writer_llm | StrOutputParser()

# critic_chain
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | critic_llm | StrOutputParser()