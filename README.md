# BEN_Task Submission: Retrieval Augmented Generation

This project contains all the code and files for the [task]().

The project is based on _Retrieval Augmented Generation (RAG)_ using _Llama-2_ model for incorporating information given in document(s) (_pdf docs_).

## Setting up the project

Create a conda environment

```shell
conda create --name venv_ben python==3.9
conda activate venv_ben
```

Install the latest version of pytorch (_CUDA 11.6_)
```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

Install all the required dependencies
```shell
pip install -r requirements.txt
```

Download the [Llama-2-7-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main) model from [Huggingface](https://huggingface.co/). The version used for this task: `llama-2-7b-chat.Q8_0.gguf`.

This is the GGUF format of the original [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf). GGUF is the format supported by [llama.cpp](https://github.com/ggerganov/llama.cpp)

Store the research paper(s) in `.pdf` format in the data folder.

## Running the project

1. Create a vector DB to store the embeddings for [Retrieval-Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401).

```shell
python build_vec_db.py
```

The generated embeddings will be stored in vectorstore folder using [FAISS](https://github.com/facebookresearch/faiss)

** I used [General Text Embeddings (GTE)](https://arxiv.org/abs/2308.03281) model `"thenlper/gte-small"` for embeddings generation from [Sentence Transformer](https://www.sbert.net/), which outperforms other embedding generation models. ([See comparison](https://huggingface.co/thenlper/gte-small))

2. Run the project demo:

```shell
python main.py --input "How does reward modeling works in Llama2-chat?"
```
Using RAG with LLama model is a two phase activity _*Retrieval*_ and _*Generation*_:
- In the retrieval phase, the program finds the relevant information based on the user’s query from the source document store _(stored vector embedding)_
- In the generation phase, the LLM uses both the retrieved information and its own knowledge to form an answer. It can also provide source links for transparency.

3. Run the project with Chatbot UI and Rest API _(Using Flask Server)_:

```shell
python api_main.py
```

** Run the UI in your browser: `https:localhost:5000`

** Use the Rest API: `https:localhost:5000/get?query=Please define the training process of the Llama model.`

Results screenshot and text files are stored in [assets]() folder.

For storing chatbot responses, I used simple file storage. It could be used with other database sources. ([link]())

4. Experiment with different chunking segmentation strategies.

```shell
python exp_text_segment.py
```







