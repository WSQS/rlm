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
    system="You are a helpful assistant.",
    messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hi, how are you?"
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


if __name__ == "__main__":
    main()
