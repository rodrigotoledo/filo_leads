import os
import yaml
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_community.llms import OpenAI, Ollama
from langchain_openai import ChatOpenAI
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

from langchain_community.llms import OpenAI, Ollama
from langchain_openai import ChatOpenAI

load_dotenv()
# Obtém as chaves da API do arquivo .env
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@staticmethod
class LLMS:
    def __init__(self):
        self.OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        self.OpenAIGPT4oMini = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.8)
        self.OpenAIGPT4o = ChatOpenAI(model_name="gpt-4o", temperature=0.8)
        self.OpenAIGPT4 = ChatOpenAI(model_name="gpt-4", temperature=0.8)
        # self.Phi3 = Ollama(model="phi3:mini")
        # self.Llama3_1 = Ollama(model="llama3.1")
        # self.Phi3 = Ollama(model="phi3:medium-128k")
        # # self.Phi3 = ChatOpenAI(model_name="phi3:medium-128k", temperature=0, api_key="ollama", base_url="http://localhost:11434")
        # self.groqLama3_8B_3192 = ChatGroq(temperature=0.5, groq_api_key=os.environ.get("GROQ_API_KEY"),
        #                                   model_name="llama3-8b-8192")





# Inicialize a instância do LLM (OpenAI) com a chave da API
openai_llm = ChatOpenAI(api_key=OPENAI_API_KEY)
search_tool = SerperDevTool(api_key=SERPER_API_KEY)
scrape_tool = ScrapeWebsiteTool()


@CrewBase
class Filo:
	"""Filo crew"""
	
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'
 
	def __init__(self):
		super().__init__()
		self.llms = LLMS()  # imported from llms.py
		self.OnlineSearchTool = SerperDevTool()
 
	@agent
	def agent_searcher_on_internet(self) -> Agent:
		return Agent(
			config=self.agents_config['agent_searcher_on_internet'],
			tools=[search_tool, scrape_tool],
			llm=self.llms.OpenAIGPT4,
			verbose=True
		)
  
	@agent
	def agent_focus_on_target(self) -> Agent:
		return Agent(
			config=self.agents_config['agent_focus_on_target'],
			tools=[search_tool, scrape_tool],
   		llm=self.llms.OpenAIGPT4,
			verbose=True
		)
	@agent
	def agent_generator(self) -> Agent:
		return Agent(
			config=self.agents_config['agent_generator'],
			tools=[search_tool, scrape_tool],
   		llm=self.llms.OpenAIGPT4,
			verbose=True
		)
  
	@task
	def task_lead_capture(self) -> Task:
		return Task(
			config=self.tasks_config['task_lead_capture'],
			output_file='task_lead_capture.html'
		)
	@task
	def task_refine_profiles(self) -> Task:
		return Task(
			config=self.tasks_config['task_refine_profiles'],
   		output_file='task_refine_profiles.html'
		)
  
	@task
	def task_new_customer(self) -> Task:
		return Task(
			config=self.tasks_config['task_new_customer'],
			output_file='task_new_customer.html'
		)
  
  
	@crew
	def crew(self) -> Crew:
		"""Creates the Filo crew"""
		return Crew(
			agents=self.agents,  # Automatically created by the @agent decorator
			tasks=self.tasks,  # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# Uncomment the following line for hierarchical processing
			# process=Process.hierarchical
		)