from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings


def run_build_db():
    loader = DirectoryLoader("../data/", glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    # The model 'thenlper/gte-small' can only handle text with a length of less than 512 (ref: Paper)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='thenlper/gte-small', model_kwargs={'device': 'cpu'})

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local("../vectorstore/db_faiss")


if __name__ == "__main__":
    run_build_db()
