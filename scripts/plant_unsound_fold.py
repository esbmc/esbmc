#!/usr/bin/env python3
"""Plant a deliberately unsound simplifier fold, or remove it again.

The equivalence check is only worth running if it would notice an unsound
rewrite, and for a year it would not: the hook saw whole expressions rather
than the per-node peepholes that actually change meaning, so a wrong fold
passed unreported (esbmc/esbmc#7260). Nothing in the suite would have caught
that narrowing, because a check that proves nothing looks exactly like a check
that proves everything.

So the CI job plants this fold and requires the run to die on it. The fold is
the one from the issue: `x * 2 -> x`, wrong for every x but zero.

Anchored on the function signature rather than kept as a context diff, so a
change to mul2t::do_simplify()'s body cannot make it apply in the wrong place
-- it either finds the signature or fails.

  plant_unsound_fold.py plant  <src/util/expr/expr_simplifier.cpp>
  plant_unsound_fold.py remove <src/util/expr/expr_simplifier.cpp>
"""

import pathlib
import sys

ANCHOR = "expr2tc mul2t::do_simplify() const\n{\n"

FOLD = """  // Planted by scripts/plant_unsound_fold.py -- x * 2 -> x is wrong for every
  // x but zero. Never commit this.
  if (is_constant_int2t(side_2) && to_constant_int2t(side_2).value == BigInt(2))
    return side_1;
"""


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in ("plant", "remove"):
        print(__doc__, file=sys.stderr)
        return 2

    action, path = sys.argv[1], pathlib.Path(sys.argv[2])
    source = path.read_text()

    if action == "plant":
        if FOLD in source:
            print(f"{path}: a fold is already planted", file=sys.stderr)
            return 1
        if source.count(ANCHOR) != 1:
            print(
                f"{path}: expected exactly one {ANCHOR.splitlines()[0]!r}, "
                f"found {source.count(ANCHOR)}",
                file=sys.stderr,
            )
            return 1
        path.write_text(source.replace(ANCHOR, ANCHOR + FOLD, 1))
        verb = "planted"
    else:
        if FOLD not in source:
            print(f"{path}: no planted fold to remove", file=sys.stderr)
            return 1
        path.write_text(source.replace(FOLD, ""))
        verb = "removed"

    print(f"{verb} the unsound fold in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
