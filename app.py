from crewai import Crew, Agent, Task, Process
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken, os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.4,
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
      description=f'Summarize the {text} clearly including every single information given. Paraphrase quotes, if any. Ensure the result is under 500 words.',
      expected_output= "Final summary of the text should include only the contents provided in the text and should be UNDER 500 WORDS"
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
    # split the original text so that the summary will be on these individual chunks instead of the whole text  
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=round(len(self.text)/3),
    chunk_overlap=round((0.3)*len(self.text)/3),
    )
    text_list = text_splitter.split_text(self.text)
    
    # saving the original text 
    text = self.text

    # encoding
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    # initialize final_output to an empty list so that each summary is appended later
    final_output = []
    
    # using i to check if its the original text or the summary appended text
    # using n to check the text_list index number
    (i, n) = (0, 0)
    
    # creating a new text_splitter for the appended summary texts
    text_splitter = RecursiveCharacterTextSplitter(
        separators=' ',
        chunk_size=5000,
        chunk_overlap=500,
        length_function=len,
          )
    
    
    while n < len(text_list):
      if len(encoding.encode(text=text)) <= 5000:                             # checking if the original text has less than 5000 tokens
        final_output.append(SummarizationCrew.agents_tasks(text=text))
        if n==0 and i==0:                                                     # if the original text has less than 5000 tokens then break from the loop
          break
        n += 1
        i = 0
        text = self.text
      elif len(encoding.encode(text_list[n])) <= 5000 and i==0:               # check if the first chunk of the text_list has less than 5000 tokens
        final_output.append(SummarizationCrew.agents_tasks(text=text_list[n]))
        n += 1
        i = 0
      else:
        if i == 0:                                                            # check if the text to be passed should be the nth chunk from text_list\
          chunks = text_splitter.split_text(text=text_list[n])                # or the appended text
        else:
          chunks = text_splitter.split_text(text=text)                        # split the chunks according to what text is passed
        summaries = []
        for chunk in chunks:
          summary = SummarizationCrew.agents_tasks(text=chunk)                # call the summary agent to each chunk
          summaries.append(summary)
        text = " ".join(summaries)
        i = 1                                                                 # initialize i to 1 so that the loop will not use text_list[n] in its code\
    return "\n\n".join(final_output)                                          # and will use only the appended text to further chunk or summarize 

