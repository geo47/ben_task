from langchain import LlamaCpp
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager


def build_llm(token_wise):
    callback_manager = None

    # For token-wise streaming
    if token_wise:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm = LlamaCpp(
        model_path="models/llama-2-7b-chat.Q8_0.gguf",
        temperature=0.01,
        max_tokens=256,
        top_p=1,
        callback_manager=callback_manager)

    return llm
