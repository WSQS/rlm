from code import InteractiveConsole
import contextlib
from dataclasses import dataclass
import io
import traceback

from anthropic import Anthropic
from dotenv import load_dotenv

class ReplInstance:
    def __init__(self):
        self.locals :dict[str,object] = {"__name__": "__console__", "__doc__": None}
        self.ic = InteractiveConsole(self.locals)
    
    @dataclass
    class RunResult:
        out : str
        err : str

    def run(self,code:str):
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                self.ic.runsource(code, "<console>", "exec")  # True means more input required
            except SystemExit:
                print("SystemExit: exit()/quit() is not allowed in this REPL session.")
            except Exception:
                print(traceback.format_exc())
        return self.RunResult(out.getvalue(),err.getvalue())



def main():
    load_dotenv()
    print("Hello from rlm!")
    ic = ReplInstance()
    r = ic.run("""print("hello,world")""")
    print(r)
    client = Anthropic()
    message = client.messages.create(
    model="MiniMax-M2.5",
    max_tokens=1000,
    system="""You are an assistant that can execute Python via a tool.

You have one tool available:

* `run_python(code: str)`: Executes the provided Python code in a **persistent** Python REPL and returns captured `stdout` and `stderr`.

Rules (must follow):

1. If you need to compute, verify, inspect data, parse text, search within content, or test an idea, you **must** call `run_python`. Do **not** paste Python code in plain text or in markdown fences.
2. Each `run_python` call must contain only the minimal code needed for the current step. If necessary, use multiple tool calls and iterate.
3. After each tool result, read `stdout` and `stderr` carefully:

   * If there is any error (or suspicious output), fix the code and call the tool again.
   * If output is long, use Python to narrow it down (filter, truncate, summarize, sample) before proceeding.
4. When you have the final answer, output **exactly one line** and nothing else:

   * `FINAL(<answer>)`
   * Do not include extra commentary, formatting, or additional lines.

Start now. Use the tool-driven workflow until you can produce `FINAL(...)`.
""",
    tools = [
    {
        "name": "run_python",
        "description": "Execute the provided Python code in a persistent REPL and return captured stdout/stderr. Use this to inspect variables, process `context`, and compute intermediate results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Do not include markdown fences."
                }
            },
            "required": ["code"],
            "additionalProperties": False
        },
    }
]
    ,
    messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Calculate 500 times 100"
                    }
            ]
            }
        ]
    )
    for block in message.content:
        if block.type == "thinking":
            print(f"Thinking:\n{block.thinking}\n")
        elif block.type == "text":
            print(f"Text:\n{block.text}\n")
        else:
            print(f"Block:\n{block}")


if __name__ == "__main__":
    main()
