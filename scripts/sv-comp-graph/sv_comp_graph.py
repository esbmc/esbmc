#!/usr/bin/env python3
"""Plot past SV-COMP Overall scores for every ESBMC configuration.

For each requested year, fetches the SV-COMP results-verified index page,
locates the Overall (or C-Overall / C.Overall) category row, records the
score for every ESBMC variant present (ESBMC, ESBMC-kind, ESBMC-incr, ...),
streams the per-tool BenchExec page to read the ESBMC version, and renders:

  * one bar chart per discovered variant (sv-comp-<variant>.png), and
  * a "best of" bar chart (sv-comp-best.png) that picks the top-scoring
    ESBMC variant in each year and labels it with the winning config.

Usage:
    pipenv install
    pipenv run python sv_comp_graph.py
    pipenv run python sv_comp_graph.py --first-year 2017 --last-year 2026 \\
        --variants esbmc esbmc-kind   # only plot a specific subset
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import requests

BASE_URL = "https://sv-comp.sosy-lab.org"
USER_AGENT = "esbmc-sv-comp-graph/1.0"

# Labels SV-COMP has used for the all-categories C-language Overall ranking.
OVERALL_LABELS = {"Overall", "C-Overall", "C.Overall"}

OVERALL_LINK_RE = re.compile(
    r"<a[^>]*href=['\"](META_[A-Za-z.-]*Overall)\.table\.html['\"][^>]*>" r"([^<]+)</a>"
)
MAX_SCORE_RE = re.compile(r"max\.\s*score:\s*(\d+)")
ESBMC_CELL_RE = re.compile(
    r"META_[A-Za-z.-]*Overall_(esbmc[a-z0-9-]*)\.table\.html['\"]>(-?\d+)"
)

# Works across BenchExec HTML styles:
#   2017-era:  <td>ESBMC ESBMC version 3.1 64-bit ...</td>
#   2020-era:  "tool": {"content": [["ESBMC version 6.1.0 ...", 5]], ...}
#   2023+:     "tool": "ESBMC", ..., "version": "version 7.7.0 64-bit ..."
# The `[^<{}]` class keeps the match inside a single HTML element/JSON object
# so it doesn't cross-match some other tool's metadata below it.
TOOL_VERSION_RE = re.compile(
    r"ESBMC[^<{}]{0,500}?version\s+(\d+(?:\.\d+)*)", re.IGNORECASE
)

# Per-tool BenchExec tables are 15-20 MB, but every format places the tool
# metadata in the first ~100 KB. Give a generous margin and stop.
VERSION_STREAM_LIMIT = 200_000

LOG = logging.getLogger("sv-comp-graph")


@dataclass(frozen=True)
class YearResult:
    year: int
    variant: str
    score: int
    max_score: int
    version: str

    @property
    def pct(self) -> float:
        return 100.0 * self.score / self.max_score


def _cache_key(url: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", url)


def fetch_index(url: str, cache_dir: Path, session: requests.Session) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{_cache_key(url)}.html"
    if cached.exists() and cached.stat().st_size > 0:
        return cached.read_text(encoding="utf-8", errors="replace")
    LOG.info("GET %s", url)
    resp = session.get(url, timeout=60, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    cached.write_text(resp.text, encoding="utf-8")
    return resp.text


def fetch_tool_version(
    url: str, cache_dir: Path, session: requests.Session
) -> str | None:
    """Stream a per-tool BenchExec page and return the ESBMC version string."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{_cache_key(url)}.version.txt"
    if cached.exists():
        text = cached.read_text(encoding="utf-8").strip()
        return text or None

    LOG.info("GET %s (streaming for version)", url)
    buf = ""
    try:
        with session.get(
            url, timeout=60, stream=True, headers={"User-Agent": USER_AGENT}
        ) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=16384):
                if not chunk:
                    continue
                buf += chunk.decode("utf-8", errors="replace")
                match = TOOL_VERSION_RE.search(buf)
                if match:
                    version = match.group(1)
                    cached.write_text(version, encoding="utf-8")
                    return version
                if len(buf) >= VERSION_STREAM_LIMIT:
                    break
    except requests.RequestException as e:
        LOG.warning("version fetch failed (%s): %s", url, e)
        return None

    LOG.warning("could not extract ESBMC version from %s", url)
    cached.write_text("", encoding="utf-8")
    return None


def split_rows(html: str) -> list[str]:
    """Slice the index HTML into <tr>…</tr>-ish segments."""
    return [seg for seg in re.split(r"(?=<tr\b)", html) if seg.startswith("<tr")]


def find_overall_row(html: str) -> tuple[str, str, int] | None:
    """Locate the per-tool ESBMC score row for Overall.

    Returns (row_html, overall_href, max_score), where overall_href is the
    per-row prefix like "META_Overall" or "META_C.Overall".
    """
    for row in split_rows(html):
        if "main score" not in row:
            continue
        m_link = OVERALL_LINK_RE.search(row)
        if not m_link:
            continue
        label = m_link.group(2).strip()
        if label not in OVERALL_LABELS:
            continue
        m_max = MAX_SCORE_RE.search(row)
        if not m_max:
            continue
        return row, m_link.group(1), int(m_max.group(1))
    return None


def extract_variant_scores(row: str) -> dict[str, int]:
    """All ESBMC-flavoured tools present in an Overall row, mapped to their score."""
    return {m.group(1): int(m.group(2)) for m in ESBMC_CELL_RE.finditer(row)}


def scrape_year(
    year: int, cache_dir: Path, session: requests.Session
) -> list[YearResult]:
    index_url = f"{BASE_URL}/{year}/results/results-verified/"
    try:
        html = fetch_index(index_url, cache_dir, session)
    except requests.RequestException as e:
        LOG.warning("[%d] index fetch failed: %s", year, e)
        return []

    found = find_overall_row(html)
    if found is None:
        LOG.warning("[%d] no Overall row found — skipping", year)
        return []
    row, overall_href, max_score = found

    scores = extract_variant_scores(row)
    if not scores:
        LOG.info("[%d] no ESBMC variants present in Overall row", year)
        return []

    out: list[YearResult] = []
    for variant, score in sorted(scores.items()):
        tool_url = (
            f"{BASE_URL}/{year}/results/results-verified/"
            f"{overall_href}_{variant}.table.html"
        )
        version = fetch_tool_version(tool_url, cache_dir, session) or ""
        LOG.info(
            "[%d] %-15s score=%d max=%d (%.1f%%) version=%s",
            year,
            variant,
            score,
            max_score,
            100.0 * score / max_score,
            version or "?",
        )
        out.append(YearResult(year, variant, score, max_score, version))
    return out


def _render_bars(
    results: list[YearResult],
    output: Path,
    title: str,
    label_fn,
) -> None:
    results = sorted(results, key=lambda r: r.year)
    xs = list(range(len(results)))
    pcts = [r.pct for r in results]
    labels = [label_fn(r) for r in results]

    fig, ax = plt.subplots(figsize=(max(8.0, 1.4 * len(results)), 5.5))
    bars = ax.bar(xs, pcts, color="#2C68A3", edgecolor="black")
    for bar, pct in zip(bars, pcts):
        ax.annotate(
            f"{pct:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, pct),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Overall score (% of max)")
    ax.set_title(title)
    top = max(pcts) if pcts else 0
    bottom = min(pcts) if pcts else 0
    ax.set_ylim(min(0.0, bottom - 5.0), max(top + 10.0, 50.0))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    LOG.info("wrote %s", output)


def plot_variant(variant: str, results: list[YearResult], output: Path) -> None:
    _render_bars(
        results,
        output,
        title=f"{variant.upper()} at SV-COMP — Overall score",
        label_fn=lambda r: (
            f"{r.year}\n({r.version})" if r.version else f"{r.year}\n(?)"
        ),
    )


def best_per_year(results: list[YearResult]) -> list[YearResult]:
    by_year: dict[int, YearResult] = {}
    for r in results:
        cur = by_year.get(r.year)
        if cur is None or r.score > cur.score:
            by_year[r.year] = r
    return [by_year[y] for y in sorted(by_year)]


def plot_best(results: list[YearResult], output: Path) -> None:
    def label(r: YearResult) -> str:
        ver = f" {r.version}" if r.version else ""
        return f"{r.year}\n({r.variant}{ver})"

    _render_bars(
        best_per_year(results),
        output,
        title="ESBMC at SV-COMP — Best variant per year",
        label_fn=label,
    )


def main(argv: list[str] | None = None) -> int:
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    # Start from 2017 since format of results is not standardized before that
    ap.add_argument("--first-year", type=int, default=2017)
    ap.add_argument("--last-year", type=int, default=2026)
    ap.add_argument(
        "--variants",
        nargs="+",
        help=(
            "ESBMC variants to plot (one image per variant). "
            "If omitted, every variant discovered in the Overall rows is plotted."
        ),
    )
    ap.add_argument(
        "--no-best",
        action="store_true",
        help="skip the sv-comp-best.png (top-variant-per-year) chart",
    )
    ap.add_argument("--cache-dir", type=Path, default=script_dir / ".cache")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="directory where sv-comp-<variant>.png files are written",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.first_year > args.last_year:
        LOG.error("--first-year must be <= --last-year")
        return 2

    all_results: list[YearResult] = []
    with requests.Session() as session:
        for year in range(args.first_year, args.last_year + 1):
            all_results.extend(scrape_year(year, args.cache_dir, session))

    if not all_results:
        LOG.error("no results scraped — nothing to plot")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    discovered = sorted({r.variant for r in all_results})
    if args.variants:
        unknown = [v for v in args.variants if v not in discovered]
        for v in unknown:
            LOG.warning("variant %s has no scraped data — skipping", v)
        variants = [v for v in args.variants if v in discovered]
    else:
        variants = discovered
    LOG.info("plotting variants: %s", ", ".join(variants) or "(none)")

    rc = 0
    for variant in variants:
        variant_results = [r for r in all_results if r.variant == variant]
        output = args.output_dir / f"sv-comp-{variant}.png"
        plot_variant(variant, variant_results, output)

    if not args.no_best:
        plot_best(all_results, args.output_dir / "sv-comp-best.png")

    return rc


if __name__ == "__main__":
    sys.exit(main())
