from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate

topic = input("Enter a topic: ")
#prompt = input("Enter your prompt: ")

llm = ChatOllama(model="deepseek-r1:1.5b", base_url="http://localhost:11434")

prompt = PromptTemplate.from_template(
    "You are a senior tourist guide in Paris. Explain {topic} clearly."
)

response = llm.invoke(prompt.format(topic=topic))
print(response.content) 