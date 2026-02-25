from code import InteractiveConsole
import contextlib
from dataclasses import dataclass
import io
import textwrap
import traceback

from anthropic import Anthropic
from anthropic.types import MessageParam, ToolResultBlockParam
from dotenv import load_dotenv


class ReplInstance:
    def __init__(self):
        self.locals: dict[str, object] = {"__name__": "__console__", "__doc__": None}
        self.ic = InteractiveConsole(self.locals)

    @dataclass
    class RunResult:
        out: str
        err: str

    def run(self, code: str):
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                self.ic.runsource(
                    code, "<console>", "exec"
                )  # True means more input required
            except SystemExit:
                print("SystemExit: exit()/quit() is not allowed in this REPL session.")
            except Exception:
                print(traceback.format_exc())
        return self.RunResult(out.getvalue(), err.getvalue())


def main():
    load_dotenv()
    print("Hello from rlm!")
    ic = ReplInstance()
    r = ic.run("""print("hello,world")""")
    print(r)
    conversation: list[MessageParam] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Calculate 500 times 100"}],
        }
    ]
    client = Anthropic()
    while True:
        message = client.messages.create(
            model="MiniMax-M2.5",
            max_tokens=2000,
            system=textwrap.dedent("""You are an iterative tool-using agent. Your job is to answer the user’s query by interacting with a persistent Python REPL via a tool, and only then produce a final answer.

Environment and tool:

* The environment contains a **persistent** Python REPL (state is preserved across calls).
* You have one tool: `run_python(code: str)` which executes Python code and returns captured `stdout` and `stderr`.
* A variable named `context` may be available in the REPL and can contain crucial information for the query. Always inspect it when it exists.

Core rules (must follow):

1. **Do not write Python code in plain text or markdown fences.** If you need any computation, verification, parsing, searching, inspection, or transformation, you **must** call `run_python`.
2. On the **first iteration** of a new task, do **not** answer immediately. Your first action should be to use `run_python` to inspect relevant data (e.g., check `context`, compute basics, or set up a plan).
3. When using `run_python`, **print key results explicitly** (e.g., `print(...)`). Do not rely on expression-only statements to show output.
4. After each tool call, carefully read both `stdout` and `stderr`:

   * If `stderr` is non-empty or output looks wrong, fix and retry with another `run_python` call.
   * Tool outputs may be truncated; if you need more, use Python to filter, sample, summarize, or narrow results before continuing.
5. Break problems into small steps and validate each step with the tool. Prefer a programmatic strategy over guessing.

Finishing:

* When you have the final answer, output **exactly one line** and nothing else:

  * `FINAL(<answer>)`
* Do not include extra commentary, explanations, or additional lines outside `FINAL(...)`.
"""),
            tools=[
                {
                    "name": "run_python",
                    "description": "Execute the provided Python code in a persistent REPL and return captured stdout/stderr. Use this to inspect variables, process `context`, and compute intermediate results.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute. Do not include markdown fences.",
                            }
                        },
                        "required": ["code"],
                        "additionalProperties": False,
                    },
                }
            ],
            messages=conversation,
        )
        conversation.append(MessageParam(role="assistant", content=message.content))
        has_tool = False
        for block in message.content:
            if block.type == "thinking":
                print(f"Thinking:\n{block.thinking}\n")
            elif block.type == "text":
                print(f"Text:\n{block.text}\n")
            elif block.type == "tool_use":
                has_tool = True
                print(f"Tool:\n{block}")
                r = ic.run(str(block.input["code"]))
                print(f"Tool Result:\n{r}")
                conversation.append(
                    MessageParam(
                        role="user",
                        content=[
                            ToolResultBlockParam(
                                type="tool_result", tool_use_id=block.id, content=str(r)
                            )
                        ],
                    )
                )
            else:
                print(f"Block:\n{block}")
        if not has_tool:
            break


if __name__ == "__main__":
    main()
