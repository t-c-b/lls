import logging

from pygls.server import LanguageServer

from lsprotocol.types import (
    TEXT_DOCUMENT_COMPLETION,
    CompletionItem,
    CompletionList,
    CompletionParams,
    InsertTextMode,
    TextEdit,
    Range,
    Position,
)

import torch

from sys import argv

from vllm import LLM, SamplingParams

logging.basicConfig(filename='log', filemode='w', level=logging.DEBUG)

server = LanguageServer("example-server", "v0.1")

llm = LLM(model="TheBloke/CodeLlama-7B-Instruct-GPTQ")
print("Model Loaded")
llm_params = SamplingParams(temperature=0.3, top_p=0.95)

@server.feature(TEXT_DOCUMENT_COMPLETION)
def completions(params: CompletionParams):
    uri = params.text_document.uri
    doc = server.workspace.get_document(uri)
    pos = params.position

    prompt = ""

    if ".chat" in uri:
        prompt = doc.source
    else:
        prefix = doc.source[:params.position.character]
        suffix = doc.source[params.position.character:]
        prompt = prefix + "<FILL_ME>" + suffix

    server.show_message(message="Infilling...")

    out = llm.generate([prompt], llm_params)[0].text

    item = CompletionItem(
            label = "llama",
            text_edit = TextEdit(
                range = Range(start=pos, end=pos),
                new_text = out,
            ),
    )

    return CompletionList(is_incomplete=False, items=[item])

server.start_io()
server.show_message("Ready.")
