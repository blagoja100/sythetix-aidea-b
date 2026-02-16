from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from retriever import Retriever
from tools.logger import Logger

logger = Logger("AgentMain")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

if __name__ == "__main__":
    retriever = Retriever()

    logger.log("Initializing LLM...")
    llm_model = "deepseek-r1:1.5b"
    llm_base_url = "http://localhost:11434"
    llm = ChatOllama(model=llm_model, base_url=llm_base_url)
    logger.log(f"LLM initialized. {llm_model} at {llm_base_url}")

    prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Use ONLY the context below to answer.
        Context:
        {context}
        Question:
        {question}Explain to 
        If the answer is not in the context, say you don't know.
        """)
    
    rag_chain = (
    {
        "question": RunnablePassthrough(),
        "context": RunnableLambda(
            lambda x: format_docs(retriever.get_relevant_documents(x["question"]))
        ),
    }
        | prompt
        | llm
        | StrOutputParser()
    )

    while True:
        topic = input("Enter a topic (or 'quit' to exit): ")
        if topic.lower() == "quit":
            break
        answer = rag_chain.invoke({"question": topic})
        print(answer)
