from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()

# Search agent: back on qwen — no browser-tool hallucination risk here
search_llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    max_tokens=900
)

# Reader agent: also qwen — same reasoning
reader_llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    max_tokens=1000
)

# Writer chain: NO tools involved, safe to use gpt-oss-20b for quality
writer_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3,
    max_tokens=4000
)

# Critic chain: NO tools involved either, qwen is fine — short output
critic_llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    max_tokens=700
)

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

# writer chain (unchanged)
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports. Every claim must be backed by specifics from the research — names, numbers, dates, or direct findings. Never write vague generalities."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction (contextualize why this topic matters, 3-4 sentences)
- Key Findings (minimum 3 points, each 4-6 sentences with concrete details from the research — not generic statements)
- Conclusion (synthesize the findings into a clear takeaway)
- Sources (list all URLs found in the research)

Be detailed, factual, and professional. If the research is limited on a point, say so explicitly rather than inventing detail."""),
])

writer_chain = writer_prompt | writer_llm | StrOutputParser()

# critic_chain (unchanged)
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