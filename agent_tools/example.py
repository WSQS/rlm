"""Example tools for demonstration."""

from agent_tools import tool


@tool(name="echo", description="Echo back the provided message")
def echo(message: str):
    """message: The message to echo back"""
    return f"Echo: {message}"


@tool(name="add", description="Add two numbers and return the result")
def add(a: int, b: int):
    """a: First number
    b: Second number"""
    return a + b
