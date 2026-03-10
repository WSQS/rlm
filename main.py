import json
from code import InteractiveConsole
import contextlib
from dataclasses import dataclass
import io
import os
import sys
import textwrap
import traceback
from datetime import datetime
from typing import Any, Dict

from anthropic import Anthropic
from anthropic.types import MessageParam, TextBlockParam, ToolResultBlockParam
from dotenv import load_dotenv


# JSONL logging setup
def get_log_file_path() -> str:
    """Get the path to the JSONL log file."""
    rlm_dir = os.path.join(os.getcwd(), ".rlm")
    os.makedirs(rlm_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(rlm_dir, f"{timestamp}.jsonl")


LOG_FILE_PATH = get_log_file_path()


def log_to_jsonl(data: Dict[str, Any]):
    """Append a JSONL log entry to the log file."""
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


DEFAULT_SYSTEM_PROMPT = textwrap.dedent("""You are an iterative tool-using agent. Your job is to answer the user's query by interacting with a persistent Python REPL via a tool, and only then produce a final answer.

Environment and tool:

* The environment contains a **persistent** Python REPL (state is preserved across calls).
* You have one tool: `run_python(code: str)` which executes Python code and returns captured `stdout` and `stderr`.
* A variable named `context` is **pre-loaded** in the REPL. This variable contains the task/query to solve. Access it directly with `print(context)` or process it in your Python code.
* **If `context` is too long to read completely**, use Python code to process it (e.g., `print(context[:200])`, `print(len(context))`, `print(context.split('\n')[0])`, etc.). **Never try to handle long context manually** - always use code.
* A function named `agent` is **pre-loaded** in the REPL. You can call `agent(new_context)` or `agent(new_context, custom_system_prompt)` to recursively invoke the agent with a new context/tasks. It will return the final answer from the sub-agent. **Use this when you encounter a gap that cannot be resolved by deterministic code logic.**

Core rules (must follow):

1. **Do not write Python code in plain text or markdown fences.** If you need any computation, verification, parsing, searching, inspection, or transformation, you **must** call `run_python`.
2. On the **first iteration** of a new task, **always start by inspecting the `context` variable** - run `print(context)` to see what needs to be solved.
3. **Execute tasks in a code-driven manner.** Break down the problem into deterministic steps and implement them directly in Python code. Only output code that can be executed.
4. **When encountering unbridgeable gaps** — situations where deterministic logic cannot resolve the problem (e.g., unknown file contents, missing information that requires external judgment, ambiguous requirements that need understanding beyond code) — use the `agent()` function to delegate that subtask to a sub-agent. The sub-agent will return its result, which you can then process further.
5. When using `run_python`, **print key results explicitly** (e.g., `print(...)`). Do not rely on expression-only statements to show output.
6. After each tool call, carefully read both `stdout` and `stderr`:

   * If the `stderr` is non-empty or output looks wrong, fix and retry with another `run_python` call.
   * Tool outputs may be truncated; if you need more, use Python to filter, sample, summarize, or narrow results before continuing.
7. Break problems into small steps and validate each step with the tool. Prefer a programmatic strategy over guessing.
**Execution Strategy:**
- First, try to solve the problem with deterministic Python code.
- If you encounter a gap that cannot be bridged by code (e.g., unknown file content, ambiguous requirements, decisions requiring external judgment), call `agent()` to delegate that subtask.
- After the sub-agent returns, continue processing with code.
**Output Guidelines:**
- Keep your output concise and relevant. Avoid printing unnecessary debug information.
- Only output what is essential to answer the user's question.
- If printing large datasets or results, use slicing (e.g., `result[:10]`) or summary methods to limit output.

Finishing:

* When you have the final answer, call `FINAL(<answer>)` in the REPL to produce your final answer.
* Do not output FINAL as text - you must call the FINAL() function using the run_python tool.
""")


class ReplInstance:
    def __init__(self):
        self.locals: dict[str, object] = {"__name__": "__console__", "__doc__": None}
        self.ic = InteractiveConsole(self.locals)
        # Add FINAL function to REPL locals
        self.final_result = None

        def FINAL(answer: object):
            self.final_result = answer

        self.locals["FINAL"] = FINAL

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


def agent(context: Any, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    ic = ReplInstance()
    ic.locals["context"] = context
    ic.locals["agent"] = agent
    conversation: list[MessageParam] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Solve the problem in context variable in REPL environment, type of context is {type(context)}, head of context is {str(context)[:100]}.",
                }
            ],
        }
    ]
    # Log initial user message
    log_to_jsonl({
        "type": "user_message",
        "content": conversation[0]["content"]
    })
    client = Anthropic()
    while True:
        message = client.messages.create(
            model="MiniMax-M2.5",
            max_tokens=2000,
            system=system_prompt,
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
        # Log assistant message
        log_to_jsonl({
            "type": "assistant_message",
            "content": [block.model_dump() for block in message.content]
        })
        conversation.append(MessageParam(role="assistant", content=message.content))
        has_tool = False
        for block in message.content:
            if block.type == "thinking":
                with contextlib.redirect_stdout(sys.stdout), contextlib.redirect_stderr(sys.stderr):
                    print(f"Thinking:\n{block.thinking}\n")
            elif block.type == "text":
                with contextlib.redirect_stdout(sys.stdout), contextlib.redirect_stderr(sys.stderr):
                    print(f"Text:\n{block.text}\n")
            elif block.type == "tool_use":
                has_tool = True
                with contextlib.redirect_stdout(sys.stdout), contextlib.redirect_stderr(sys.stderr):
                    print(f"Tool:\n{block}")
                r = ic.run(str(block.input["code"]))
                # Truncate stdout and stderr separately, then return as JSON
                max_tool_result_length = 2000  # Limit each output length
                stdout = r.out
                stderr = r.err
                if len(stdout) > max_tool_result_length:
                    stdout = (
                        stdout[:max_tool_result_length] + "\n... [output truncated]"
                    )
                if len(stderr) > max_tool_result_length:
                    stderr = (
                        stderr[:max_tool_result_length] + "\n... [output truncated]"
                    )
                tool_result_str = json.dumps({"stdout": stdout, "stderr": stderr})
                with contextlib.redirect_stdout(sys.stdout), contextlib.redirect_stderr(sys.stderr):
                    print(f"Tool Result:\n{tool_result_str}")
                # Log tool result
                log_to_jsonl({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "result": {"stdout": stdout, "stderr": stderr}
                })
                conversation.append(
                    MessageParam(
                        role="user",
                        content=[
                            ToolResultBlockParam(
                                type="tool_result",
                                tool_use_id=block.id,
                                content=tool_result_str,
                            )
                        ],
                    )
                )
            else:
                with contextlib.redirect_stdout(sys.stdout), contextlib.redirect_stderr(sys.stderr):
                    print(f"Block:\n{block}")

        if ic.final_result:
            with contextlib.redirect_stdout(sys.stdout), contextlib.redirect_stderr(sys.stderr):
                print(f"FINAL:\n{ic.final_result}")
            # Log final result
            log_to_jsonl({
                "type": "final_result",
                "result": ic.final_result
            })
            return ic.final_result

        if not has_tool:
            with contextlib.redirect_stdout(sys.stdout), contextlib.redirect_stderr(sys.stderr):
                print("No result and no tool. Add user message.")
            conversation.append(
                MessageParam(
                    role="user",
                    content=[
                        TextBlockParam(
                            type="text",
                            text="""You did not produce a tool call or a FINAL result.
Do not answer in plain text.
Continue from your last step by calling run_python, and when finished, output the final answer via FINAL(...).""",
                        )
                    ],
                )
            )


def main():
    load_dotenv()
    print("Hello from rlm!")
    print(f"Logging to: {LOG_FILE_PATH}")
    context = "Calculate 500 times 100"
    agent(context)


if __name__ == "__main__":
    main()
