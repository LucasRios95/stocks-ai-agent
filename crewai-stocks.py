# IMPORT DAS LIBS
import json
import os
from datetime import datetime

import yfinance as yf

from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults


import streamlit as st

# CRIANDO FUNÇÃO YAHOO FINANCE TOOL


def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2023-08-08", end="2024-08-20")
    return stock


yahoo_finance_tool = Tool(
    name="Yahoo Finance Tool",
    description="Fetches stock prices for {ticket} from the last year about a specific stock from Yahoo Finance API",
    func=lambda ticket: fetch_stock_price(ticket)
)

# IMPORTANDO OPENAI LLM GPT
os.environ['OPENAI_API_KEY'] = st.secrets['OPEN_API_KEY']
llm = ChatOpenAI(model="gpt-3.5-turbo")

stockPriceAnalyst = Agent(
    role="Senior stock price Analyst",
    goal="find the {ticket} stock price and analyses trends",
    backstory="""You're a highly experienced in analyzing the price of an specific stock
    and make predicitions about its future price
""",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    tools=[yahoo_finance_tool],
    allow_dellegation=False
)

getStockPrice = Task(
    description="Analyze the stock {ticket} price history and creat a trend analyses of up, down ou sideways",
    expected_output="""Specify the current trend stock price - up, down or sideways.
    eg. stock= 'AAPL, price UP'
""",
    agent=stockPriceAnalyst
)

# IMPORTANDO A TOOL DE SEARCH
search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)

newsAnalyst = Agent(
    role="Stock News Analyst",
    goal="""Create a short summary of the market news related to stock {ticket} company. Specify a trend -up, down or sidewys 
    with the news content
""",
    backstory=""" You're highly experienced in analyzing the market trends and news and have tracked assets for more than 10 years. 

    You're also master level analyst in the tradicional markets and have deep understanding of human psychology. 

    You understand news, theirs titles and information but you look at those with a health dose of skepticism. You consider also, the source of the news 
""",
    verbose=True,
    llm=llm,
    max_iter=10,
    memory=True,
    tools=[search_tool],
    allow_dellegation=False
)

get_news = Task(
    description=f"""Take the stock and always include BTC to it (if not requested).
    Use the search tool to search each one individually.

    The current date is {datetime.now()}

    Compose the results into a helpfull report
""",
    expected_output=""" A sumary off the overall market and one sentence summary for each requested asset.
    Include a fear/greed score for each asset based on the news. Use format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE> 
""",
    agent=stockPriceAnalyst
)

stockAnalystWriter = Agent(
    role="Senior stock analyst writer",
    goal="""Analyse the trends price and news and write an insightfull compelling and informative 3 paragraph long newsletter based on the stock report and price trend.
""",
    backstory="""You're widely accepted as the best stock analyst in the market. You understand complex concepts and create compelling stories
    and narratives that resonate with wider audiences.

    You understand macro factors and combine multiple theories - eg. cycle theory and fundamental analyses.
    You're able to hold multiple opinions when analyzing anything 
""",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    allow_dellegation=True
)

writeAnalyses = Task(
    description="""Use stock price trend and the news stock report to create an analyses and write a newsletter about the {ticket} company
    that is brief and highlights the most important points.
    Focus on the stock price trends, news and fear/greed score. What are the near future considerations?
    Include the previous analyses of stock trends and news summary.  
""",
    expected_output=""" An eloquent 3 paragraphs newsletter formated as markdown in an easy readable manner. It should contain:
     
    - 3 bullet executive summary
    - Introduction - set the overall pictures and spike up the interest
    - main part provides the meat of the analysis including the news summary and fear/greed scores
    - summary - key facts and future trend predictions - up, down or sideways.
""",
    agent=stockAnalystWriter,
    context=[getStockPrice, get_news]
)

crew = Crew(
    agents=[stockPriceAnalyst, newsAnalyst, stockAnalystWriter],
    tasks=[getStockPrice, get_news, writeAnalyses],
    verbose=2,
    process=Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15
)

with st.sidebar:
    st.header('Enter the ticket stock')

    with st.form(key='research form'):
        topic = st.text_input("select the ticket")
        submit_button = st.form_submit_button(label="Run research")

if submit_button:
    if not topic:
        st.error("Please fill the ticket field")
    else:
        results = crew.kickoff(inputs={'ticket': topic})

        st.subheader("Results of your research:")
        st.write(results['final_output'])
