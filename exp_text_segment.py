from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter, \
    NLTKTextSplitter


def run_char_segment(documents):

    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_splitter_sep = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator='\n'
    )
    texts = text_splitter.split_documents(documents)
    texts_sep = text_splitter_sep.split_documents(documents)

    print("====   Sample chunks using CharacterTextSplitter without 'separator':   ====\n\n")
    for i, chunk in enumerate(texts):
        if i < 4:
            print(f"### Chunk {i + 1}: \n{chunk.page_content}\n")

    print("====   Sample chunks using CharacterTextSplitter with 'separator':   ====\n\n")
    for i, chunk in enumerate(texts_sep):
        if i < 4:
            print(f"### Chunk {i + 1}: \n{chunk.page_content}\n")


def run_rec_char_segment(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_splitter_sep = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=['\n']
    )
    texts = text_splitter.split_documents(documents)
    texts_sep = text_splitter_sep.split_documents(documents)

    print("====   Sample chunks using RecursiveCharacterTextSplitter without 'separator':   ====\n\n")
    for i, chunk in enumerate(texts):
        if i < 4:
            print(f"### Chunk {i + 1}: \n{chunk.page_content}\n")

    print("====   Sample chunks using RecursiveCharacterTextSplitter with 'separator':   ====\n\n")
    for i, chunk in enumerate(texts_sep):
        if i < 4:
            print(f"### Chunk {i + 1}: \n{chunk.page_content}\n")


def run_tok_segment(documents):

    text_splitter = TokenTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    texts = text_splitter.split_documents(documents)

    print("====   Token-based Chunking   ====\n\n")
    for i, chunk in enumerate(texts):
        if i < 4:
            print(f"### Chunk {i + 1}: \n{chunk.page_content}\n")


def run_context_aware_segment(documents):

    text_splitter = NLTKTextSplitter()

    texts = text_splitter.split_documents(documents)

    print("====   Context-Aware chunking   ====\n\n")
    for i, chunk in enumerate(texts):
        if i < 4:
            print(f"### Chunk {i + 1}: \n{chunk.page_content}\n")


if __name__ == "__main__":
    loader = DirectoryLoader("data/", glob='*.pdf', loader_cls=PyPDFLoader)
    docs = loader.load()

    """
    Different chunking strategies with standard chunk size: 500
    
    The model used for embedding text chunks 'thenlper/gte-small' can only handle
    text with a length of less than 512 (ref: Paper).
    """
    run_char_segment(docs)
    run_rec_char_segment(docs)
    run_tok_segment(docs)
    run_context_aware_segment(docs)
