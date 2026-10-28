"""Load text documents from a folder."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_SUFFIXES = {".txt", ".md"}


def clean_markdown(text: str) -> str:
    """Attach each markdown heading to the paragraph below it.

    "## Sharing dependencies" followed by "Developers record ..." becomes the single line
    "Sharing dependencies: Developers record ...". That keeps the heading's keywords in the
    same sentence, which helps retrieval and makes answers easier to read.
    """
    lines: list[str] = []
    pending: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip().rstrip(".:")
            if pending:  
                lines.append(pending + ".")
            pending = heading or None
        elif stripped and pending:
            lines.append(f"{pending}: {stripped}")
            pending = None
        else:
            lines.append(line)
    if pending:
        lines.append(pending + ".")
    return "\n".join(lines)


def load_documents(folder: str | Path) -> dict[str, str]:
    """Return {file name: text} for every .txt / .md file in `folder` (non-recursive)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Documents folder not found: {folder}")

    docs = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            text = clean_markdown(raw).strip() if path.suffix.lower() == ".md" else raw.strip()
            if text:
                docs[path.name] = text
    if not docs:
        raise ValueError(f"No .txt or .md documents found in {folder}")
    return docs
