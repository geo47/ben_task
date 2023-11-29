import argparse
from pprint import pprint
from time import time

from src.chain_pipeline import setup_chain

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        type=str,
                        default="How does reward modeling works in Llama2-chat?",
                        help="Enter the query to pass into the LLM")
    args = parser.parse_args()

    start = time()
    chain = setup_chain()
    response = chain({"query": args.input})
    end = time()

    print(f"\nAnswer: {response['result']}")
    print("=" * 50)

    pprint(response)

    # Source document
    source_docs = response["source_documents"]
    for i, doc in enumerate(source_docs):
        print(f"\nSource Document {i+1}\n")
        print(f"Source Text: {doc.page_content}")
        print(f"Document Name: {doc.metadata['source']}")
        print(f"Page Number: {doc.metadata['page']}\n")
        print("=" * 60)

    print(f"Time to retrieve response: {end - start} seconds.")
