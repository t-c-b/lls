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
        revision="gptq-4bit-32g-actorder_True"
)
tokenizer = AutoTokenizer.from_pretrained(src, use_fast=True)

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
        #prefix = ''.join(doc.lines[:params.position.line])
        #suffix = ''.join(doc.lines[params.position.line:])

        #prompt = f"<PRE> {prefix} <SUF> {suffix} <MID>"
        prompt = prefix + "<FILL_ME>" + suffix

    server.show_message(message="Infilling...")

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    output = model.generate(
        inputs=input_ids,
        temperature=0.2,
        do_sample=True,
        top_p=0.9,
        top_k=40,
        max_new_tokens=512,
        repetition_penalty=1.3,
    )
    #out = tokenizer.decode(output[0])
    out = tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
    #out = out[out.find("<MID>")+5 : out.find("<EOT>")]
    logging.info(out)

    lines = out.splitlines()

    item = CompletionItem(
            label = "llama",
            #insert_text = out,
            #insert_text_mode = InsertTextMode.AsIs,
            text_edit = TextEdit(
                range = Range(start=pos, end=pos),
                new_text = out,
            ),
    )

    return CompletionList(is_incomplete=False, items=[item])

server.start_io()
