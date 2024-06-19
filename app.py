from crewai import Agent, Task
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken, os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("OPENAI_API_KEY"),
)

class SummarizationCrew():
  def __init__(self, text: str):
    self.text = text

  def agents_tasks(text):
    agent = Agent(
      role='Principal Researcher',
      goal='Do amazing research and summaries on the {text} you are working with. Do not report anything verbatim',
      backstory="You're a Principal Summarizer at a big company and you need to summarize about a given topic.",
      llm=llm,
      verbose=True,
      allow_delegation=False
      )
    task = Task(
      agent=agent,
      description=f'Analyze and summarize the {text} below, make sure to include the most relevant information in the summary, return only the summary nothing else.\
      Do not include information that is not important. Ensure you summarize the quotes and do not report anything verbatim.\n\nCONTENT\n----------\n{text}',
      expected_output= "Summary of the document"
      )

    summary = task.execute()
    return summary

  def run(self):

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=round(len(self.text)/4),
    chunk_overlap=0,
    )
    text_list = text_splitter.split_text(self.text)

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    final_output = []
    n = 0
    while n < len(text_list):
      if len(encoding.encode(text_list[n])) <= 5000:
        final_output.append(SummarizationCrew.agents_tasks(text=text_list[n]))
        n += 1
      else:
        text_splitter = RecursiveCharacterTextSplitter(
        separators=' ',
        chunk_size=5000,
        chunk_overlap=500,
        length_function=len,
          )
        chunks = text_splitter.split_text(text=text_list[n])
        summaries = []
        for chunk in chunks:
          summary = SummarizationCrew.agents_tasks(text=chunk)
          summaries.append(summary)
        text = " ".join(summaries)
    return "\n\n".join(final_output)
