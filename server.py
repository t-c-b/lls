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
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sys import argv

logging.basicConfig(filename='log', filemode='w', level=logging.DEBUG)

src = "TheBloke/CodeLlama-7B-Instruct-GPTQ"

server = LanguageServer("example-server", "v0.1")
model = AutoModelForCausalLM.from_pretrained(
        src, 
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        revision="gptq-4bit-32g-actorder_True",
)
tokenizer = AutoTokenizer.from_pretrained(src, use_fast=True)

@server.feature(TEXT_DOCUMENT_COMPLETION)
def completions(params: CompletionParams):
    uri = params.text_document.uri
    doc = server.workspace.get_document(uri)

    line = params.position.line
    char = params.position.character

    prompt = ""

    if ".chat" in uri:
        prompt = doc.source
    else:
        prefix = ''.join(doc.lines[:line])
        prefix = prefix + doc.lines[line][:char]

        suffix = ''.join(doc.lines[line+1:])
        suffix = doc.lines[line][char:] + suffix

        # This model does not use CodeLlamaTokenizer so manually prompt infill
        prompt = f"<PRE> {prefix} <SUF> {suffix} <MID>"
        logging.info(prompt)

    server.show_message(message="Infilling...")

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    output = model.generate(
        inputs=input_ids,
        temperature=0.3,
        do_sample=True,
        top_p=0.9,
        top_k=40,
        max_new_tokens=256,
        repetition_penalty=1.1,
    )

    out = tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
    out = out[:out.find("<EOT>")]

    logging.info(out)

    lines = out.splitlines()

    item = CompletionItem(
            label = "llama",
            insert_text = out,
            insert_text_mode = InsertTextMode.AsIs,
    )

    return CompletionList(is_incomplete=False, items=[item])

server.start_io()
