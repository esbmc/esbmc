#!/usr/bin/env python3
"""Build GNU info manual from the Hugo markdown docs in website/content/docs/.

Dependencies: pandoc, texinfo (makeinfo), pyyaml
"""

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Page:
    path: Path
    title: str
    weight: int
    body: str


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from content. Returns (metadata, body)."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta = yaml.safe_load(parts[1]) or {}
    return meta, parts[2].lstrip("\n")


def load_page(path: Path) -> Page:
    """Load a markdown file into a Page."""
    text = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(text)
    title = str(meta.get("title", path.stem))
    weight = int(meta.get("weight", 999))
    return Page(path=path, title=title, weight=weight, body=body)


def collect_pages(directory: Path) -> list[Page]:
    """Find all .md files (excluding _index.md) in directory, sorted by weight."""
    pages = []
    for f in directory.iterdir():
        if f.is_file() and f.suffix == ".md" and f.name != "_index.md":
            pages.append(load_page(f))
    pages.sort(key=lambda p: p.weight)
    return pages


def collect_subsections(directory: Path) -> list[tuple[int, Path]]:
    """Find child directories with _index.md, return sorted by weight."""
    subs = []
    for d in directory.iterdir():
        if d.is_dir():
            idx = d / "_index.md"
            if idx.exists():
                meta, _ = parse_frontmatter(idx.read_text(encoding="utf-8"))
                weight = int(meta.get("weight", 999))
                subs.append((weight, d))
    subs.sort(key=lambda x: x[0])
    return subs


def build_node_map(docs_dir: Path) -> dict[str, str]:
    """Build mapping from relative doc path to page title for link rewriting.

    Maps paths like '/docs/function-contracts' to 'Function Contracts'.
    """
    node_map: dict[str, str] = {}

    for md_file in docs_dir.rglob("*.md"):
        meta, _ = parse_frontmatter(md_file.read_text(encoding="utf-8"))
        title = meta.get("title")
        if not title:
            continue
        # Build the Hugo URL path from the file path
        rel = md_file.relative_to(docs_dir.parent)  # relative to website/content
        if md_file.name == "_index.md":
            url_path = "/" + str(rel.parent).replace("\\", "/")
        else:
            url_path = "/" + str(rel.with_suffix("")).replace("\\", "/")
        node_map[url_path] = str(title)

    return node_map


def shift_headings(text: str, level: int) -> str:
    """Shift all markdown headings by (level - 1) hashes, skipping fenced code blocks."""
    if level <= 1:
        return text
    shift = level - 1
    prefix = "#" * shift
    lines = text.split("\n")
    result = []
    in_fence = False
    for line in lines:
        if re.match(r"^(`{3,}|~{3,})", line):
            in_fence = not in_fence
        if not in_fence and re.match(r"^#{1,6} ", line):
            line = prefix + line
        result.append(line)
    return "\n".join(result)


def slugify(title: str) -> str:
    """Convert a title to a pandoc-style slug for anchor links."""
    s = title.lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s]+", "-", s).strip("-")
    return s


def rewrite_links(text: str, node_map: dict[str, str]) -> str:
    """Rewrite inter-page links to anchor references for pandoc."""

    def replace_link(m: re.Match) -> str:
        link_text = m.group(1)
        url_path = m.group(2)
        if url_path in node_map:
            node_title = node_map[url_path]
            slug = slugify(node_title)
            return f"[{link_text}](#{slug})"
        return m.group(0)

    return re.sub(r"\[([^\]]+)\]\((/docs/[^)]+)\)", replace_link, text)


def strip_shortcodes(text: str) -> str:
    """Remove Hugo shortcodes and <script> blocks."""
    # Remove {{< ... >}} shortcodes (opening/closing/self-closing)
    text = re.sub(r"\{\{<[^>]*>\}\}", "", text)
    # Remove {{% ... %}} shortcodes
    text = re.sub(r"\{\{%[^%]*%\}\}", "", text)
    # Remove <script>...</script> blocks (including multi-line)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
    return text


def emit_page(page: Page, level: int, node_map: dict[str, str]) -> str:
    """Emit a page: title as heading at level, body with shifted headings."""
    heading = "#" * level + " " + page.title
    body = shift_headings(page.body, level)
    body = rewrite_links(body, node_map)
    return f"\n{heading}\n\n{body}\n"


def emit_section(
    directory: Path, level: int, docs_dir: Path, node_map: dict[str, str]
) -> str:
    """Emit _index.md at level, then child pages and sub-sections at level+1."""
    parts: list[str] = []

    idx = directory / "_index.md"
    if idx.exists():
        page = load_page(idx)
        parts.append(emit_page(page, level, node_map))

    for page in collect_pages(directory):
        parts.append(emit_page(page, level + 1, node_map))

    for _, subdir in collect_subsections(directory):
        parts.append(emit_section(subdir, level + 1, docs_dir, node_map))

    return "".join(parts)


HEADER = """\
---
title: ESBMC User Manual
---

ESBMC (Efficient SMT-based Context-Bounded Model Checker) automatically
detects or proves the absence of runtime errors in C, C++, CUDA, CHERI,
Kotlin, Python, and Solidity programs using SMT solvers.

Thank you for using ESBMC. ESBMC is made possible by the following
organizations:

The ESBMC development was supported by various research funding agencies,
including CNPq (Brazil), CAPES (Brazil), FAPEAM (Brazil), EPSRC (UK), Royal
Society (UK), British Council (UK), European Commission (Horizon 2020), and
companies including Intel, Nokia Institute of Technology, Samsung, and Veribee.
ESBMC is currently funded by Intel, EPSRC grants EP/T026995/1, EP/V000497/1,
EU H2020 ELEGANT 957286 and Soteria project awarded by the UK Research and
Innovation for the Digital Security by Design (DSbD) Programme.

The S3 team and by extension ESBMC, is sponsored by Zulip. Zulip is an
organized team chat app designed for efficient communication.

# ESBMC User Manual

"""


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(
        description="Build GNU info manual from Hugo markdown docs."
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=str(repo_root / "build"),
        help="directory for output files (default: <repo>/build)",
    )
    parser.add_argument(
        "--docs-dir",
        default=str(repo_root / "website" / "content" / "docs"),
        help="path to Hugo docs directory (default: website/content/docs)",
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    out_dir = Path(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build node map for cross-page link rewriting
    node_map = build_node_map(docs_dir)

    # Assemble combined markdown
    parts: list[str] = [HEADER]

    # Top-level pages
    for page in collect_pages(docs_dir):
        parts.append(emit_page(page, 2, node_map))

    # Top-level sections
    for _, subdir in collect_subsections(docs_dir):
        parts.append(emit_section(subdir, 2, docs_dir, node_map))

    combined_text = strip_shortcodes("".join(parts))

    combined = out_dir / "esbmc-manual.md"
    combined.write_text(combined_text, encoding="utf-8")
    print(f"Combined markdown: {combined}")

    # Convert to texinfo
    texi = out_dir / "esbmc.texi"
    subprocess.run(
        [
            "pandoc",
            "-f",
            "markdown",
            "-t",
            "texinfo",
            "--standalone",
            "-V",
            "title=ESBMC User Manual",
            "-V",
            "author=ESBMC Developers",
            str(combined),
            "-o",
            str(texi),
        ],
        check=True,
    )
    print(f"Texinfo source: {texi}")

    # Build info file
    info = out_dir / "esbmc.info"
    subprocess.run(
        ["makeinfo", "--no-split", "--force", str(texi), "-o", str(info)],
        check=True,
    )
    print(f"GNU info manual: {info}")


if __name__ == "__main__":
    main()
