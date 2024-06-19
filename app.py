from crewai import Crew, Agent, Task, Process
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken, os
from dotenv import load_dotenv

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
    summary_agent = Agent(
      role='Principal Summarizer',
      goal='Do summaries on the {text} you are working with. Do not report anything verbatim. DO NOT report any information not provided in the text.',
      backstory="You're a Principal Summarizer at a big company and you need to summarize the given text.",
      llm=llm,
      verbose=True,
      allow_delegation=False,
      memory=True
      )
    summary_task = Task(
      agent=summary_agent,
      description=f'Summarize only the {text} given, make sure to include the most relevant information given in the text in the summary, return only the summary nothing else.\
      DO NOT make up information. DO NOT repeat information. Ensure you summarize the quotes and do not report anything verbatim.\
      Do not report any information not present in the {text}. Dates and locations mentioned in the {text} are important, do include it in the report.\
      Do not mention if something is not present in the text. Ensure you are gramatically correct\n\nCONTENT\n----------\n{text}',
      expected_output= "Final summary of the text should include only the contents provided in the text"
      )
    
    verify_agent = Agent(
      role='Principal Grammar Checker',
      goal='Verify if the contents produced by the {summary_agent} are gramatically correct. Do not report anything verbatim. DO NOT make up information.',
      backstory="You're an expert Principal Grammar verfier at a big company and you need to verify the grammar in a given text.",
      llm=llm,
      verbose=True,
      allow_delegation=True,
      memory=False
      )

    verify_task = Task(
      agent=verify_agent,
      description=f'Verify if the {text} provided and the contents produced by the summary agent is gramatically correct. If not then call the summary agent to do its task again.\
      Ensure that the contents produced are gramatically correct. DO NOT make up information. DO NOT report anything not presented by the summary agent.',
      expected_output="Verified summary of the text provided"
      )
    
    crew = Crew(
      agents=[summary_agent],
      tasks=[summary_task],
      process=Process.sequential
    )
    summary = crew.kickoff()
    return summary

  def run(self):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=round(len(self.text)/2),
    chunk_overlap=0,
    )
    text_list = text_splitter.split_text(self.text)
    text = self.text

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    final_output = []
    i, n = (0, 0)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=' ',
        chunk_size=5000,
        chunk_overlap=500,
        length_function=len,
          )
    while n < len(text_list):
      if len(encoding.encode(text_list[n])) <= 5000 and i==0:
        final_output.append(SummarizationCrew.agents_tasks(text=text_list[n]))
        n += 1
        i = 0
      elif len(encoding.encode(text=text)) <= 5000:
        final_output.append(SummarizationCrew.agents_tasks(text=text))
        n += 1
        i = 0
      else:
        if i == 0:
          chunks = text_splitter.split_text(text=text_list[n])
        else:
          chunks = text_splitter.split_text(text=text) 
        summaries = []
        for chunk in chunks:
          summary = SummarizationCrew.agents_tasks(text=chunk)
          summaries.append(summary)
        text = " ".join(summaries)
        i = 1
    return "\n\n".join(final_output)


