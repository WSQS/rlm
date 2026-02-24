from code import InteractiveConsole
import contextlib
from dataclasses import dataclass
import io
import traceback

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
    print("Hello from rlm!")
    ic = ReplInstance()
    r = ic.run("""print("hello,world")""")
    print(r)


if __name__ == "__main__":
    main()
