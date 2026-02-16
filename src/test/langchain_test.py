from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from opensearchpy import OpenSearch

# Initialize OpenSearch client
client = OpenSearch(
    hosts=[{'host': 'localhost', 'port': 9200}],
    http_compress=True,
    http_auth=('admin', 'admin'),
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False
)

index_name = "langchain-test-index"
index_body = {
    "settings": {
        "index": {
            "number_of_shards": 1
        }
    }
}

if not client.indices.exists(index=index_name):
    client.indices.create(index=index_name, body=index_body)
    print(f"Created index: {index_name}")

topic = input("Enter a topic: ")
#prompt = input("Enter your prompt: ")

llm = ChatOllama(model="deepseek-r1:1.5b", base_url="http://localhost:11434")

prompt = PromptTemplate.from_template(
    "You are a senior tourist guide in Paris. Explain {topic} clearly."
)

response = llm.invoke(prompt.format(topic=topic))
print(response.content) 