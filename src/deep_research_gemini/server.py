"""MCP server for Gemini Deep Research Agent."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

from google import genai
from mcp.server.fastmcp import FastMCP

# Globals set by CLI args
_format_instructions: Optional[str] = None
_report_dir: Path = Path("./transient")


def _extract_report_text(interaction) -> str:
    """Extract final report text from google-genai interaction shapes."""
    # google-genai < 2 exposed final text as interaction.outputs[-1].text.
    outputs = getattr(interaction, "outputs", None)
    if outputs:
        text = getattr(outputs[-1], "text", None)
        if text:
            return text

    # google-genai >= 2 exposes final text as the last model_output step.
    for step in reversed(getattr(interaction, "steps", None) or []):
        if getattr(step, "type", None) != "model_output":
            continue
        texts = [getattr(item, "text", "") for item in (getattr(step, "content", None) or [])]
        report_text = "\n".join(text for text in texts if text)
        if report_text:
            return report_text

    raise RuntimeError(f"Research completed but no report text was found: {interaction!r}")


def _run_research(query: str) -> str:
    """Execute deep research and return the report text."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")

    warnings.filterwarnings(
        "ignore",
        message=".*Interactions usage is experimental.*",
        category=UserWarning,
    )

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
            return _extract_report_text(interaction)
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


mcp = FastMCP("deep-research-gemini", log_level="WARNING")


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


def main() -> None:
    global _format_instructions, _report_dir

    # Quiet noisy MCP INFO logs like "Processing request of type ..."
    logging.getLogger("mcp").setLevel(logging.WARNING)
    logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Gemini Deep Research CLI and MCP server")
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
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Run as an MCP server instead of CLI filter mode",
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

    if args.mcp:
        mcp.run()
        return

    query = sys.stdin.read().strip()
    if not query:
        parser.error("requires a query on stdin; use --mcp for MCP server mode")
    report_text = _run_research(query)
    sys.stdout.write(report_text)
    if not report_text.endswith("\n"):
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
