#!/usr/bin/env python3
"""Plot historical ESBMC SV-COMP scores from git history of stat files.

For each month in the requested range, finds the commit in
scripts/competitions/svcomp/stats-<budget>{.txt,} whose author date is
nearest to the middle of that month, extracts that file version via
`git show`, parses the `Score: N (max: M)` line, and renders:

  * one line graph per time budget (sv-comp-hist-<budget>.png), and
  * a multi-line graph with all budgets overlaid (sv-comp-hist-combined.png).

Usage:
    pipenv install
    pipenv run python sv_comp_history.py
    pipenv run python sv_comp_history.py --first-year 2023 --last-year 2026 \\
        --budgets 30s 300s
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BUDGETS: dict[str, str] = {
    "30s": "scripts/competitions/svcomp/stats-30s.txt",
    "100s": "scripts/competitions/svcomp/stats-100s",
    "300s": "scripts/competitions/svcomp/stats-300s.txt",
    "600s": "scripts/competitions/svcomp/stats-600s.txt",
}
BUDGET_ORDER = sorted(BUDGETS.keys(), key=lambda b: int(b.rstrip("s")))

SCORE_RE = re.compile(r"Score:\s+(\d+)\s+\(max:\s+(\d+)\)")

LOG = logging.getLogger("sv-comp-history")


@dataclass(frozen=True)
class HistPoint:
    year: int
    month: int
    budget: str
    commit: str
    commit_date: datetime
    score: int
    max_score: int

    @property
    def pct(self) -> float:
        return 100.0 * self.score / self.max_score

    @property
    def x(self) -> datetime:
        return datetime(self.year, self.month, 15, tzinfo=timezone.utc)


def git_log_for_path(path: str, repo: Path) -> list[tuple[str, datetime]]:
    out = subprocess.run(
        ["git", "log", "--format=%H %aI", "--", path],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    commits: list[tuple[str, datetime]] = []
    for line in out.splitlines():
        sha, _, iso = line.partition(" ")
        if sha and iso:
            commits.append((sha, datetime.fromisoformat(iso)))
    return commits


def pick_commit_for_month(
    commits: list[tuple[str, datetime]],
    year: int,
    month: int,
    max_days: int = 15,
) -> tuple[str, datetime] | None:
    target = datetime(year, month, 15, tzinfo=timezone.utc)
    if not commits:
        return None
    best = min(commits, key=lambda c: abs(c[1] - target))
    if abs(best[1] - target).days > max_days:
        return None
    return best


def extract_file(commit: str, path: str, cache_dir: Path, repo: Path) -> str:
    cache = cache_dir / f"{commit[:12]}-{Path(path).name}"
    if cache.exists():
        LOG.debug("cache hit %s", cache)
        return cache.read_text(encoding="utf-8")
    out = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    cache.write_text(out, encoding="utf-8")
    return out


def parse_score(text: str) -> tuple[int, int] | None:
    m = SCORE_RE.search(text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def iter_year_months(
    first_year: int, first_month: int, last_year: int, last_month: int
) -> Iterator[tuple[int, int]]:
    y, m = first_year, first_month
    while (y, m) <= (last_year, last_month):
        yield y, m
        m += 1
        if m == 13:
            m = 1
            y += 1


def collect(
    first_year: int,
    first_month: int,
    last_year: int,
    last_month: int,
    budgets: list[str],
    cache_dir: Path,
    repo: Path,
) -> list[HistPoint]:
    points: list[HistPoint] = []
    for budget in budgets:
        path = BUDGETS[budget]
        commits = git_log_for_path(path, repo)
        if not commits:
            LOG.warning("%s: no git history", path)
            continue
        for year, month in iter_year_months(
            first_year, first_month, last_year, last_month
        ):
            picked = pick_commit_for_month(commits, year, month)
            if picked is None:
                continue
            sha, when = picked
            text = extract_file(sha, path, cache_dir, repo)
            parsed = parse_score(text)
            if parsed is None:
                LOG.warning(
                    "[%04d-%02d] %s: could not parse Score in %s",
                    year,
                    month,
                    budget,
                    sha[:12],
                )
                continue
            score, max_score = parsed
            LOG.info(
                "[%04d-%02d] %-5s commit=%s date=%s score=%d max=%d (%.1f%%)",
                year,
                month,
                budget,
                sha[:12],
                when.date().isoformat(),
                score,
                max_score,
                100.0 * score / max_score,
            )
            points.append(
                HistPoint(year, month, budget, sha[:12], when, score, max_score)
            )
    return points


def _format_time_axis(ax, points: list[HistPoint]) -> None:
    xs = [p.x for p in points]
    span_days = (max(xs) - min(xs)).days if xs else 0
    if span_days > 365 * 2:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(20)
        lbl.set_ha("right")


def _apply_yscale(ax, ys: list[float], scale: str) -> None:
    if not ys:
        return
    if scale == "log":
        ax.set_yscale("log")
        lo = max(min(ys) * 0.8, 1e-3)
        ax.set_ylim(lo, max(ys) * 1.15)
        ax.yaxis.set_major_locator(
            mticker.LogLocator(subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))
        )
        fmt = mticker.ScalarFormatter()
        fmt.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        ax.set_ylim(min(0.0, min(ys) - 5.0), max(ys) + 10.0)


def plot_budget(
    budget: str, points: list[HistPoint], output: Path, scale: str
) -> None:
    points = sorted(points, key=lambda p: p.x)
    xs = [p.x for p in points]
    ys = [p.pct for p in points]

    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.plot(xs, ys, marker="o", color="#2C68A3", linewidth=1.8, markersize=5)
    ax.set_ylabel("Overall score (% of max)")
    ax.set_title(f"ESBMC SV-COMP history (budget {budget})")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    _format_time_axis(ax, points)
    _apply_yscale(ax, ys, scale)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    LOG.info("wrote %s", output)


def plot_combined(points: list[HistPoint], output: Path, scale: str) -> None:
    if not points:
        return
    budgets = sorted({p.budget for p in points}, key=lambda b: int(b.rstrip("s")))
    colors = {"30s": "#2C68A3", "100s": "#E8871E", "300s": "#2CA02C", "600s": "#D62728"}

    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    for budget in budgets:
        bp = sorted([p for p in points if p.budget == budget], key=lambda p: p.x)
        if not bp:
            continue
        ax.plot(
            [p.x for p in bp],
            [p.pct for p in bp],
            marker="o",
            label=budget,
            color=colors.get(budget),
            linewidth=1.8,
            markersize=4,
        )

    ax.set_ylabel("Overall score (% of max)")
    ax.set_title("ESBMC SV-COMP history — all time budgets")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(title="Time budget", loc="lower right")
    _format_time_axis(ax, points)
    _apply_yscale(ax, [p.pct for p in points], scale)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    LOG.info("wrote %s", output)


def _parse_ym(s: str, default_month: int) -> tuple[int, int]:
    """Accept either 'YYYY' or 'YYYY-MM'."""
    if "-" in s:
        d = datetime.strptime(s, "%Y-%m")
        return d.year, d.month
    return int(s), default_month


def main(argv: list[str] | None = None) -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    now = datetime.now(timezone.utc)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--first",
        default="2022",
        help="start, as YYYY or YYYY-MM (default 2022, i.e. 2022-01)",
    )
    ap.add_argument(
        "--last",
        default=f"{now.year}-{now.month:02d}",
        help="end, as YYYY or YYYY-MM (default current month)",
    )
    ap.add_argument(
        "--budgets",
        nargs="+",
        choices=BUDGET_ORDER,
        default=BUDGET_ORDER,
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="directory for sv-comp-hist-*.png files",
    )
    ap.add_argument("--cache-dir", type=Path, default=Path("/tmp/esbmc-svcomp-hist"))
    ap.add_argument(
        "--scale",
        choices=("log", "linear"),
        default="log",
        help="y-axis scale (default: log)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    try:
        first_year, first_month = _parse_ym(args.first, default_month=1)
        last_year, last_month = _parse_ym(args.last, default_month=12)
    except ValueError as e:
        LOG.error("bad --first/--last: %s", e)
        return 2

    if (first_year, first_month) > (last_year, last_month):
        LOG.error("--first must be <= --last")
        return 2

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    points = collect(
        first_year,
        first_month,
        last_year,
        last_month,
        list(args.budgets),
        args.cache_dir,
        repo_root,
    )
    if not points:
        LOG.error("no data points collected — nothing to plot")
        return 1

    for budget in args.budgets:
        bp = [p for p in points if p.budget == budget]
        if not bp:
            continue
        output = args.output_dir / f"sv-comp-hist-{budget}.png"
        plot_budget(budget, bp, output, args.scale)

    plot_combined(points, args.output_dir / "sv-comp-hist-combined.png", args.scale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
