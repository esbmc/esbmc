#!/usr/bin/env python3
"""Report the IREP2 migration bars per frontend (docs/roadmap/frontends-to-irep2.md).

The bars in that document are defined as greps, and both over-count:

  B-1  counts every identifier containing `exprt`/`typet`/`codet`/`irept`, including
       the ones inside comments and doc blocks.
  B-2  counts a symbol-table write by the *spelling* of its argument, so
       `set_type(migrate_type(t))` and `set_type(some_type2tc)` are both counted as
       non-IREP2 debt when neither is.

Both raw and refined figures are printed, because the raw ones are what the document
quotes historically -- and note that those are `git grep -c` *line* counts, not
occurrence counts, which is a third way the quoted numbers differ from the bar's
wording. Both columns are given. The refinement is syntactic: it strips // and /* */ comments and
string literals, and treats a write as IREP2 when its argument is a migrate_* call, a
*2tc constructor, or a name ending in `2tc`/`2t`. It recognises three shapes of IREP2
argument: a `migrate_*` call, a `*2tc` constructor or a call to a method whose name ends
in `2t`/`2tc`, and a name declared as `expr2tc`/`type2tc` somewhere in the same file, whether
written bare or as a field reached through it.

The last is what makes a converted site stop counting: a conversion usually names its
result `value2`, `body` or similar, and the bare name keeps the grep matching. Matching
per file rather than per scope is deliberately coarse -- a name declared IREP2 in one
function and legacy in another would be misread -- so the refined figure is still an
upper bound on nothing and a *lower* bound on nothing: it is the best syntactic answer,
and `--list` prints what it counted so a disagreement can be checked by hand.
"""
import re
import subprocess
import sys

FRONTENDS = [
    "clang-c-frontend",
    "clang-cpp-frontend",
    "solidity-frontend",
    "python-frontend",
    "jimple-frontend",
]

LEGACY = re.compile(r"\b([A-Za-z_]*(?:exprt|typet|codet)|irept)\b")
WRITE = re.compile(r"\.set_(type|value)\(|->set_(type|value)\(")
IREP2_ARG = re.compile(r"^(migrate_type|migrate_expr)\s*\(|^[A-Za-z_]\w*2tc\s*\(|2tc$|2t$")
# A call whose method name ends in `2t`/`2tc` returns IREP2, e.g. `to_code2t(...)`.
IREP2_CALL = re.compile(r"^[A-Za-z_][\w.>\-\[\]]*(?:2t|2tc)\s*\(")
# Names declared as an IREP2 container anywhere in the file. Converting a site
# typically names its result `value2`, `body`, `values2` and so on, which keeps
# the grep matching; this is what makes such a site stop counting as debt.
IREP2_DECL = re.compile(r"\b(?:const\s+)?(?:expr2tc|type2tc)\s*>?\s*&?\s*([A-Za-z_]\w*)"
                        r"\s*[;=,){]")
# A reference to an IREP2 node: every field of one is itself IREP2, so
# `code_type.arguments[i]` needs no migration. The same holds through an IREP2
# container's `operator->`, so `callee->type` counts once `callee` is declared
# `expr2tc` -- which is why the two name sets are searched together below.
IREP2_NODE = re.compile(r"\b(?:const\s+)?[A-Za-z_]\w*2t\s*&\s*([A-Za-z_]\w*)\s*[;=]")
# The root of a field, element or member access, so `arguments[0]`, `x->type`
# and `code_type.arguments` are all tested against the name they start from.
BASE_NAME = re.compile(r"[.\[(>-]")
# IREP2 builders whose names do not end in `2t`/`2tc`. Only the ones with no
# legacy namesake: `gen_zero`, `gen_one` and `gen_nondet` are declared for both
# `typet` and `type2tc` (util/expr/expr_util.h, irep2/irep2_utils.h), so their
# spelling cannot say which was called and they keep counting.
IREP2_HELPER = re.compile(r"^(?:gen_true_expr|gen_false_expr|gen_long|gen_ulong)\s*\(")


def strip_noise(text):
    """Remove block comments, line comments and string literals."""
    # Keep the line count: `--list` reports file line numbers, and collapsing a
    # block comment to nothing would shift every line after it.
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    out = []
    for line in text.split("\n"):
        line = re.sub(r'"(?:[^"\\]|\\.)*"', '""', line)
        line = re.sub(r"//.*$", "", line)
        out.append(line)
    return "\n".join(out)


def argument_of(line, start):
    """The text between the parentheses of the call starting at `start`."""
    i = line.index("(", start)
    depth = 0
    for j in range(i, len(line)):
        if line[j] == "(":
            depth += 1
        elif line[j] == ")":
            depth -= 1
            if depth == 0:
                return line[i + 1:j].strip()
    return line[i + 1:].strip()


def statements(clean):
    """Yield (line map, joined text) for each statement.

    The map is (offset, line number) pairs, because neither end of a join is the
    right answer: a write split over several lines has no argument on the line
    its call starts, and a brace-less `if (...)` does not terminate a statement,
    so the write sits several lines after the one the join began on.
    """
    buf, lines, depth = "", [], 0
    for lineno, line in enumerate(clean.split("\n"), 1):
        if buf:
            buf += " "
        lines.append((len(buf), lineno))
        buf += line.strip()
        depth = max(0, depth + line.count("(") - line.count(")"))
        if depth == 0 and re.search(r"[;{}]\s*$", buf):
            yield lines, buf
            buf, lines = "", []
    if buf:
        yield lines, buf


def line_of(lines, offset):
    """The source line an offset into a joined statement came from."""
    result = lines[0][1]
    for start, lineno in lines:
        if start > offset:
            break
        result = lineno
    return result


def files(frontend):
    out = subprocess.run(["git", "ls-files", "src/" + frontend],
                         capture_output=True,
                         text=True,
                         check=True).stdout
    return [f for f in out.split() if f.endswith((".cpp", ".h"))]


def is_irep2(arg, irep2_names):
    """Whether the argument of a symbol-table write is already IREP2."""
    if IREP2_ARG.search(arg) or IREP2_CALL.match(arg) or IREP2_HELPER.match(arg):
        return True
    return BASE_NAME.split(arg)[0] in irep2_names


def measure(frontend, listing=None):
    lines_b1 = raw_b1 = refined_b1 = raw_b2 = refined_b2 = 0
    for path in files(frontend):
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        raw_b1 += len(LEGACY.findall(text))
        lines_b1 += sum(1 for ln in text.split("\n") if LEGACY.search(ln))
        clean = strip_noise(text)
        refined_b1 += len(LEGACY.findall(clean))
        irep2_names = set(IREP2_DECL.findall(clean)) | set(IREP2_NODE.findall(clean))
        for lines, line in statements(clean):
            for m in WRITE.finditer(line):
                raw_b2 += 1
                arg = argument_of(line, m.start())
                if is_irep2(arg, irep2_names):
                    continue
                refined_b2 += 1
                if listing is not None:
                    listing.append("%s:%d: %s" % (path, line_of(lines, m.start()), arg))
    return lines_b1, raw_b1, refined_b1, raw_b2, refined_b2


def main():
    print("%-22s %8s %8s %8s   %6s %6s" % ("frontend", "B-1 ln", "B-1", "B-1*", "B-2", "B-2*"))
    print("-" * 68)
    listing = [] if "--list" in sys.argv else None
    totals = [0, 0, 0, 0, 0]
    for f in FRONTENDS:
        got = measure(f, listing)
        totals = [a + b for a, b in zip(totals, got)]
        print("%-22s %8d %8d %8d   %6d %6d" % (f, *got))
    print("-" * 68)
    print("%-22s %8d %8d %8d   %6d %6d" % ("total", *totals))
    print("\n B-1 ln  matching lines, which is what `git grep -c` reports and what the")
    print("         roadmap's historical figures are.")
    print(" B-1     occurrences, which is what the bar's wording describes.")
    print(" *       refined: comments and string literals excluded from B-1; writes")
    print("         whose argument is already IREP2 excluded from B-2.")
    if listing is not None:
        print("\nB-2* sites:")
        for row in listing:
            print("  " + row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
