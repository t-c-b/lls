# LLS - Large Language Server
Language Server Protocol implementation with local LLM Inference.

## Installation
Gather the requirements `pip install -r requirements.txt`

## Usage
Configure your editor to connect to the server and use the standard completion 
functionality. The default behavior is to fill-in code, but in a file matching 
`*.chat*` it will use the file as the prompt verbatim.
