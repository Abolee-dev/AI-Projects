"""
ORION — terminal chatbot.

Usage:
    python -m cli.bot
    orion                       # after pip install -e .

Commands inside the chat:
    /ingest [dir]               ingest PDFs (default: data/documents/)
    /ingest-file <path>         ingest a single PDF
    /insights                   show regulatory topic insights
    /actions on|off             toggle action suggestions
    /clear                      clear chat history
    /help                       show commands
    /quit  |  Ctrl-C            exit
"""
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import print as rprint

from container import get_ingestion_pipeline, get_rag_chain, get_vector_store, get_bm25, get_llm
from models.schemas import IngestRequest, InsightRequest, QueryRequest
from utils.config import settings
from utils.logger import logger

console = Console()


def _banner() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]ORION[/bold cyan] — Regulatory AI Assistant\n"
            "[dim]Type your question, or /help for commands[/dim]",
            border_style="cyan",
        )
    )


def _print_answer(response) -> None:
    console.print("\n[bold green]Answer[/bold green]")
    console.print(Markdown(response.answer))

    if response.citations:
        console.print("\n[bold yellow]Citations[/bold yellow]")
        tbl = Table(show_header=True, header_style="bold")
        tbl.add_column("Source", style="cyan")
        tbl.add_column("Page", justify="right")
        tbl.add_column("Score", justify="right")
        for c in response.citations:
            tbl.add_row(c.source, str(c.page), f"{c.relevance_score:.3f}")
        console.print(tbl)

    if response.suggested_action:
        a = response.suggested_action
        console.print(
            f"\n[bold magenta]Suggested action[/bold magenta]: "
            f"{a.action_type.value} (confidence {a.confidence:.0%})"
        )
        console.print(f"  [dim]Payload: {a.payload}[/dim]")

    console.print(f"\n[dim]Latency: {response.latency_ms:.0f} ms[/dim]\n")


async def _handle_ingest(args: list[str]) -> None:
    pipeline = get_ingestion_pipeline()
    if args:
        req = IngestRequest(source_dir=args[0])
    else:
        req = IngestRequest(source_dir=str(settings.documents_dir))
    console.print("[dim]Ingesting…[/dim]")
    result = await pipeline.ingest(req)
    console.print(
        f"[green]Done[/green] — {result.processed} file(s), "
        f"{result.chunks_created} chunks ({result.duration_seconds}s)"
    )
    if result.errors:
        for e in result.errors:
            console.print(f"[red]Error:[/red] {e}")


async def _handle_ingest_file(args: list[str]) -> None:
    if not args:
        console.print("[red]Usage:[/red] /ingest-file <path>")
        return
    pipeline = get_ingestion_pipeline()
    req = IngestRequest(file_paths=[args[0]])
    result = await pipeline.ingest(req)
    console.print(
        f"[green]Done[/green] — {result.chunks_created} chunks ({result.duration_seconds}s)"
    )


async def _handle_insights() -> None:
    from insights.engine import InsightEngine
    engine = InsightEngine(get_vector_store(), get_bm25(), get_llm())
    result = await engine.generate(InsightRequest(limit=10))
    console.print("\n[bold cyan]Insight Summary[/bold cyan]")
    console.print(Markdown(result.summary))
    tbl = Table(show_header=True, header_style="bold")
    tbl.add_column("Topic", style="cyan")
    tbl.add_column("Mentions", justify="right")
    for t in result.topics:
        tbl.add_row(t.topic, str(t.count))
    console.print(tbl)


def _help() -> None:
    console.print(
        Panel(
            "/ingest [dir]          Ingest PDFs from a directory\n"
            "/ingest-file <path>    Ingest a single PDF\n"
            "/insights              Show regulatory topic insights\n"
            "/actions on|off        Toggle action suggestions\n"
            "/clear                 Clear the screen\n"
            "/help                  Show this message\n"
            "/quit                  Exit",
            title="Commands",
            border_style="dim",
        )
    )


async def _run() -> None:
    settings.ensure_dirs()
    _banner()

    chain = get_rag_chain()
    actions_enabled = False

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        parts = user_input.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("/quit", "/exit", "/q"):
            console.print("[dim]Goodbye.[/dim]")
            break
        elif cmd == "/help":
            _help()
        elif cmd == "/clear":
            console.clear()
            _banner()
        elif cmd == "/ingest":
            await _handle_ingest(args)
        elif cmd == "/ingest-file":
            await _handle_ingest_file(args)
        elif cmd == "/insights":
            await _handle_insights()
        elif cmd == "/actions":
            if args and args[0] == "on":
                actions_enabled = True
                console.print("[green]Action suggestions enabled.[/green]")
            elif args and args[0] == "off":
                actions_enabled = False
                console.print("[yellow]Action suggestions disabled.[/yellow]")
            else:
                console.print(f"Actions: {'on' if actions_enabled else 'off'}")
        else:
            # Treat as a query
            vs = get_vector_store()
            if vs.size == 0:
                console.print(
                    "[yellow]No documents indexed. Run [bold]/ingest[/bold] first.[/yellow]"
                )
                continue
            with console.status("[dim]Thinking…[/dim]", spinner="dots"):
                response = await chain.run(
                    QueryRequest(
                        query=user_input,
                        top_k=5,
                        enable_actions=actions_enabled,
                    )
                )
            _print_answer(response)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
