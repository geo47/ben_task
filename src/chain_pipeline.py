from pprint import pprint
from time import time

from langchain import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory, MongoDBChatMessageHistory
from langchain.vectorstores import FAISS
from src.prompts import qa_template
from src.llm import build_llm
from langchain.memory.chat_message_histories import FileChatMessageHistory


def set_qa_prompt():
    prompt = PromptTemplate(template=qa_template,
                            input_variables=["context", "question"])
    return prompt


def build_retrieval_qa(llm, prompt, vectordb, memory):
    # Heck for using 'return_source_documents=True' along with memory
    # https://github.com/langchain-ai/langchain/issues/2256#issuecomment-1665188576
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=vectordb.as_retriever(search_kwargs={'k': 2}),
                                        return_source_documents=True,
                                        chain_type_kwargs={'prompt': prompt},
                                        # combine_docs_chain_kwargs={'prompt': prompt},
                                        memory=memory)
    return chain


def setup_memory_db():
    # connection_string = "mongodb://admin:admin:27017"
    # message_history = MongoDBChatMessageHistory(connection_string=connection_string, session_id="test-session")
    message_history = FileChatMessageHistory(file_path="data/chat_db")
    memory = ConversationBufferMemory(memory_key="chat_history",
                                      chat_memory=message_history)
    return memory


def setup_chain(token_wise=False):
    embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small", model_kwargs={'device': 'cpu'})
    vectordb = FAISS.load_local("vectorstore/db_faiss", embeddings)
    memory = setup_memory_db()
    llm = build_llm(token_wise)
    qa_prompt = set_qa_prompt()
    chain = build_retrieval_qa(llm, qa_prompt, vectordb, memory)

    return chain


class ChainPipeline:
    def __init__(self):
        self.chain = setup_chain()

    def get_chain(self):
        return self.chain

    def run(self, query):
        start = time()
        response = self.chain({"query": query})
        end = time()
        infer_time = (end - start)
        res = {
            "result": response["result"],
            "infer_time": infer_time
        }
        return res


