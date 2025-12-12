"""MCP server for Gemini Deep Research Agent."""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from google import genai
from mcp.server.fastmcp import FastMCP

# Globals set by CLI args
_format_instructions: Optional[str] = None
_report_dir: Path = Path("./transient")


def _run_research(query: str) -> str:
    """Execute deep research and return the report text."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")

    client = genai.Client(api_key=api_key)

    full_query = query
    if _format_instructions:
        full_query = f"{query}\n\n{_format_instructions}"

    interaction = client.interactions.create(
        input=full_query,
        agent="deep-research-pro-preview-12-2025",
        background=True,
    )

    while True:
        interaction = client.interactions.get(interaction.id)
        if interaction.status == "completed":
            return interaction.outputs[-1].text
        elif interaction.status == "failed":
            raise RuntimeError(f"Research failed: {interaction.error}")
        time.sleep(10)


def _save_report(content: str, report_name: str) -> Path:
    """Save report to file and return the path."""
    _report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}-{report_name}.txt"
    filepath = _report_dir / filename
    filepath.write_text(content)
    return filepath


mcp = FastMCP("deep-research-gemini")


@mcp.tool()
def deep_research(query: str, report_name: str) -> str:
    """
    Run a deep research task using Gemini Deep Research Agent.

    This tool autonomously plans, executes, and synthesizes multi-step research
    tasks. It navigates complex information landscapes using web search to
    produce detailed, cited reports. Research typically takes several minutes.

    Args:
        query: The research question or task to investigate.
        report_name: A slug for the output filename (e.g., "ev-battery-landscape").
                     The file will be saved as {timestamp}-{report_name}.txt

    Returns:
        Path to the generated report file.
    """
    report_text = _run_research(query)
    filepath = _save_report(report_text, report_name)
    return f"Report generated: {filepath}"


def main():
    global _format_instructions, _report_dir

    parser = argparse.ArgumentParser(description="MCP server for Gemini Deep Research")
    parser.add_argument(
        "--format-instructions",
        type=Path,
        help="Path to file containing format instructions to append to queries",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("./transient"),
        help="Directory to save reports (default: ./transient/)",
    )
    args = parser.parse_args()

    if args.format_instructions:
        if args.format_instructions.exists():
            _format_instructions = args.format_instructions.read_text()
        else:
            raise FileNotFoundError(
                f"Format instructions file not found: {args.format_instructions}"
            )

    _report_dir = args.report_dir

    mcp.run()


if __name__ == "__main__":
    main()
