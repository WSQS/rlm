# RLM

This repository is an experiment in building AI workflows.

The code in this repo is inspired by [rlms](https://github.com/alexzhang13/rlm).

Programmatic Tool Calling (PTC) and related techniques allow LLMs to emit executable code and run it in a REPL environment.

Recursive Language Models (RLMs) go one step further. The model can recursively call itself to handle long contexts.

## Reflection

When handling long contexts, the LLM behaves like a parser: semantics can flow both up and down in a tree structure. From this perspective, the key point in this structure is how the semantics flow up and down. In RLMs, semantics flow down by query with sub-context, semantics flow up by subagent return it's result.

For context or system prompt, is it possible or useful to not write the situation, but just put the core code into it? I think this will like lisp, which core implementation can be written in on page. 

For RLMs, variables act as signifiers, while the context supplies the signified meanings they refer to.

Consider the situation where LLMs only output code, what they are really outputting is an abstract syntax tree (AST).
