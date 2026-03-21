"""Tool registry for automatic tool registration."""

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Any, Callable


class ToolRegistry:
    """Registry for managing tools with automatic discovery and registration."""

    def __init__(self):
        self._tools: list[dict[str, Any]] = []

    def register(self, name: str, description: str, func: Callable[..., Any]) -> Callable[..., Any]:
        """Register a tool with the registry."""
        input_schema = self._generate_input_schema(func)
        tool_def: dict[str, Any] = {
            "name": name,
            "description": description,
            "input_schema": input_schema,
            "function": func,
        }
        self._tools.append(tool_def)
        return func

    def _generate_input_schema(self, func: Callable[..., Any]) -> dict[str, Any]:
        """Generate input_schema from function signature and docstring."""
        sig = inspect.signature(func)
        properties: dict[str, Any] = {}
        required: list[str] = []

        # Parse docstring for parameter descriptions
        doc_params: dict[str, str] = {}
        if func.__doc__:
            lines = func.__doc__.split("\n")
            for line in lines:
                line = line.strip()
                if ":" in line:
                    key, desc = line.split(":", 1)
                    doc_params[key.strip()] = desc.strip()

        for param_name, param in sig.parameters.items():
            if param_name.startswith("_"):
                continue
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                if param.annotation in (int, "int"):
                    param_type = "integer"
                elif param.annotation in (float, "float"):
                    param_type = "number"
                elif param.annotation in (bool, "bool"):
                    param_type = "boolean"
                elif param.annotation in (list, "list"):
                    param_type = "array"
                elif param.annotation in (dict, "dict"):
                    param_type = "object"

            description: str = doc_params.get(param_name, f"Parameter {param_name}")
            properties[param_name] = {
                "type": param_type,
                "description": description,
            }
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def get_tools(self) -> list[dict[str, Any]]:
        """Get all registered tools as Anthropic-format tool definitions with functions."""
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["input_schema"],
                "function": tool["function"],
            }
            for tool in self._tools
        ]

    def clear(self):
        """Clear all registered tools."""
        self._tools.clear()


# Global registry instance
_registry = ToolRegistry()


def tool(name: str, description: str):
    """Decorator to register a function as a tool.

    Usage:
        @tool(name="my_tool", description="Does something")
        def my_tool(param1: str, param2: int):
            \"\"\"param1: Description for param1
            param2: Description for param2\"\"\"
            ...
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        _registry.register(name, description, func)
        return func
    return decorator


def discover_tools(package_path: str | None = None) -> None:
    """Auto-discover and load all tools from the tools package.

    Args:
        package_path: Path to the tools package. Defaults to the 'tools' folder.
    """
    if package_path is None:
        # Default to the tools folder relative to this file
        tools_path: Path = Path(__file__).parent
    else:
        tools_path = Path(package_path)

    if not tools_path.exists():
        print(f"Warning: tools directory not found at {tools_path}")
        return

    # Iterate through all Python files in the tools directory
    for module_info in pkgutil.iter_modules([str(tools_path)]):
        module_name = module_info.name
        if module_name.startswith("_"):
            continue

        # Import the module to trigger @tool decorators
        full_module_name = f"tools.{module_name}"
        try:
            importlib.import_module(full_module_name)
        except Exception as e:
            print(f"Warning: Failed to import tool module {full_module_name}: {e}")


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return _registry


def get_tools() -> list[dict[str, Any]]:
    """Get all registered tools with functions."""
    return _registry.get_tools()
