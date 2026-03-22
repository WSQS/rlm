#!/usr/bin/env python3
"""
Log Visualizer for RLM

Converts JSONL log files produced by RLM into Markdown documents for visualization.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any


def convert_entry(entry: Dict[str, Any]) -> str:
    """Convert a single log entry to Markdown."""
    type_ = entry.get('type')
    
    if type_ == 'user_message':
        content_blocks = entry.get('content', [])
        md = "# User Message\n\n"
        for block in content_blocks:
            block_type = block.get('type')
            if block_type == 'text':
                text = block.get('text', '')
                md += f"{text}\n\n"
        return md
    
    elif type_ == 'assistant_message':
        content_blocks = entry.get('content', [])
        md = ""
        for block in content_blocks:
            block_type = block.get('type')
            if block_type == 'text':
                text = block.get('text', '')
                md += f"## Assistant Text\n\n{text}\n\n"
            elif block_type == 'tool_use':
                name = block.get('name', '')
                input_ = block.get('input', {})
                md += f"## Tool Call: {name}\n\n"
                if 'code' in input_:
                    md += f"```python\n{input_['code']}\n```\n\n"
                else:
                    input_str = json.dumps(input_, indent=2)
                    md += f"```json\n{input_str}\n```\n\n"
            elif block_type == 'thinking':
                thinking = block.get('thinking', '')
                md += f"## Thinking\n\n{thinking}\n\n"
        return md
    
    elif type_ == 'tool_result':
        result = entry.get('result', {})
        stdout = result.get('stdout', '')
        stderr = result.get('stderr', '')
        md = "## Tool Result\n\n"
        if stdout:
            md += f"**Stdout:**\n\n```\n{stdout}\n```\n\n"
        if stderr:
            md += f"**Stderr:**\n\n```\n{stderr}\n```\n\n"
        return md
    
    elif type_ == 'final_result':
        result = entry.get('result', '')
        return f"# Final Result\n\n{result}\n\n"
    
    else:
        return f"## Unknown Entry Type: {type_}\n\n```json\n{json.dumps(entry, indent=2)}\n```\n\n"


def main():
    parser = argparse.ArgumentParser(description='Convert RLM JSONL log to Markdown')
    parser.add_argument('log_file', help='Path to the JSONL log file')
    args = parser.parse_args()
    
    log_path = args.log_file
    if not os.path.exists(log_path):
        print(f"Error: File {log_path} does not exist.", file=sys.stderr)
        sys.exit(1)
    
    md_path = log_path.replace('.jsonl', '.md')
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except IOError as e:
        print(f"Error reading file {log_path}: {e}", file=sys.stderr)
        sys.exit(1)
    
    md_content = ""
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            md_content += convert_entry(entry)
        except json.JSONDecodeError as e:
            md_content += f"## JSON Parse Error at line {line_num}\n\n{e}\n\nLine content: {line}\n\n"
    
    try:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
    except IOError as e:
        print(f"Error writing to {md_path}: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Markdown written to {md_path}")


if __name__ == '__main__':
    main()