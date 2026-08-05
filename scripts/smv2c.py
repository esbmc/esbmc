#!/usr/bin/env python3
"""Translate a NuSMV model into C so ESBMC can check its properties.

Written for the models the Seccom intent-decomposition framework generates
from device specifications: enumerated state variables, an ASSIGN block of
init/next case expressions, a DEFINE block of derived predicates, and
INVARSPEC / LTLSPEC / SPEC properties.

WHAT IS DELIBERATELY DROPPED, AND WHY IT MATTERS
------------------------------------------------
NuSMV's `INVAR p` and `TRANS p` are *constraints*: they shrink the model's
state space to the part satisfying p. C has no equivalent -- a program's
reachable states follow from its statements, and nothing can declare a state
unreachable by fiat.

That asymmetry is the point of this converter rather than a limitation of it.
The Seccom models contain

    INVAR !violation;        -- constrain violating states out of existence
    INVARSPEC !violation;    -- then check that violating states do not exist
    LTLSPEC G (!violation);
    SPEC AG (!violation);

so the property is assumed and then checked, and passes for that reason. The C
translation carries the transition structure but not the constraint, which asks
the question NuSMV was never asked: can the state machines actually reach a
violating state?

  * ESBMC finds one  -> the NuSMV PASS came from the constraint, and here is
                        the counterexample trace.
  * ESBMC finds none -> the property is genuinely true of the machines and the
                        INVAR was merely redundant.

Dropped constraints are listed in a comment at the top of the generated C, so
the omission is visible in the artefact rather than only here.

SCOPE: enumerated variables only; `case` bodies that are a single value or a
`{a, b, ...}` nondeterministic set; boolean DEFINE expressions over `=`, `!`,
`&`, `|`. Integer arithmetic, modules, arrays and fairness are not handled and
raise instead of being silently mistranslated.

Usage:
  smv2c.py model.smv -o model.c [--steps N] [--property NAME]
"""

import argparse
import re
import sys
from pathlib import Path


class Unsupported(Exception):
    pass


class SmvModel:
    def __init__(self):
        self.vars = {}        # name -> [values]
        self.defines = []     # (name, expr) in source order
        self.init = {}        # name -> value
        self.next = {}        # name -> [(guard_expr, [values])]
        self.invarspec = []
        self.ltlspec = []
        self.ctlspec = []
        self.constraints = [] # ("INVAR"|"TRANS", expr) -- dropped, but recorded


def _strip_comments(text):
    return "\n".join(re.sub(r"--.*$", "", ln) for ln in text.splitlines())


def parse(text: str) -> SmvModel:
    m = SmvModel()
    text = _strip_comments(text)

    for name, body in re.findall(r"^\s*(\w+)\s*:\s*\{([^}]*)\}\s*;", text, re.M):
        m.vars[name] = [v.strip() for v in body.split(",") if v.strip()]

    for name, expr in re.findall(r"^\s*(\w+)\s*:=\s*(.+?);\s*$", text, re.M):
        if name.startswith("init(") or name.startswith("next("):
            continue
        if re.fullmatch(r"\d+", expr.strip()):
            continue  # generated_feature_count and friends: not predicates
        m.defines.append((name, expr.strip()))

    for name, val in re.findall(r"init\((\w+)\)\s*:=\s*([\w]+)\s*;", text):
        m.init[name] = val

    for name, body in re.findall(r"next\((\w+)\)\s*:=\s*case(.*?)esac\s*;", text, re.S):
        cases = []
        for guard, result in re.findall(r"([^:;]+?)\s*:\s*([^;]+?)\s*;", body):
            guard = guard.strip()
            result = result.strip()
            if result.startswith("{"):
                vals = [v.strip() for v in result.strip("{}").split(",")]
            else:
                vals = [result]
            cases.append((guard, vals))
        m.next[name] = cases

    for kw, target in (("INVARSPEC", m.invarspec), ("LTLSPEC", m.ltlspec),
                       ("SPEC", m.ctlspec)):
        for expr in re.findall(rf"^\s*{kw}\s+(.+?);\s*$", text, re.M):
            target.append(expr.strip())

    for kw in ("INVAR", "TRANS"):
        for expr in re.findall(rf"^\s*{kw}\s+(.+?);\s*$", text, re.M):
            m.constraints.append((kw, expr.strip()))

    return m


def qualify(var: str, literal: str) -> str:
    """C enum constants share one namespace; SMV scopes them per variable.

    The Seccom models reuse literals freely -- s3_violation_detected appears in
    several machines -- so every constant is prefixed with its variable.
    """
    return f"{var}__{literal}"


def expr_to_c(e: str, model: SmvModel) -> str:
    """SMV boolean expression -> C. Enumerated equality becomes ==."""
    if re.search(r"[+\-*/]|\bmod\b|\bnext\s*\(", e):
        raise Unsupported(f"arithmetic or next() in expression: {e}")

    def eq(m):
        var, lit = m.group(1), m.group(2)
        if var in model.vars:
            if lit not in model.vars[var]:
                raise Unsupported(f"{lit} is not a value of {var}")
            return f"{var} == {qualify(var, lit)}"
        return m.group(0)

    out = re.sub(r"\b(\w+)\s*=\s*(\w+)\b", eq, e)
    out = re.sub(r"\bTRUE\b", "1", out)
    out = re.sub(r"\bFALSE\b", "0", out)
    out = out.replace("&", "&&").replace("|", "||")
    return out


def generate(model: SmvModel, steps: int, source: str,
             only_property=None) -> str:
    fsm = sorted(model.next)                       # driven by next()
    free = [v for v in sorted(model.vars) if v not in model.next]

    L = []
    a = L.append
    a("/* Generated by scripts/smv2c.py from %s -- do not edit. */" % source)
    a("/*")
    a(" * NuSMV constraints are NOT represented, because C cannot express them:")
    a(" * a program's reachable states follow from its statements. The model's")
    a(" * constraints were:")
    if model.constraints:
        for kw, expr in model.constraints:
            a(f" *     {kw} {expr};")
        a(" *")
        a(" * Any INVAR that matches a checked property means the property was")
        a(" * assumed and then verified. Here it is only checked.")
    else:
        a(" *     (none)")
    a(" */")
    a("#include <assert.h>")
    a("")
    a("extern unsigned int __VERIFIER_nondet_uint(void);")
    a("void __ESBMC_assume(_Bool);")
    a("")
    a("static unsigned int pick(unsigned int n)")
    a("{")
    a("  unsigned int c = __VERIFIER_nondet_uint();")
    a("  __ESBMC_assume(c < n);")
    a("  return c;")
    a("}")
    a("")

    for v in sorted(model.vars):
        a("enum %s_e { %s };" % (
            v, ", ".join(qualify(v, x) for x in model.vars[v])))
    a("")
    for v in sorted(model.vars):
        a("static enum %s_e %s;" % (v, v))
    a("")

    a("static void step_free(void)")
    a("{")
    for v in free:
        a("  %s = (enum %s_e)pick(%d);" % (v, v, len(model.vars[v])))
    a("}")
    a("")

    a("static void step_fsm(void)")
    a("{")
    for v in fsm:
        a("  {")
        a("    enum %s_e nx = %s;" % (v, v))
        first = True
        for guard, vals in model.next[v]:
            g = expr_to_c(guard, model)
            kw = "if" if first else "else if"
            first = False
            if g in ("1",):
                a("    else {")
            else:
                a("    %s (%s) {" % (kw, g))
            if len(vals) == 1:
                a("      nx = %s;" % qualify(v, vals[0]))
            else:
                a("      switch (pick(%d)) {" % len(vals))
                for i, val in enumerate(vals):
                    a("      case %d: nx = %s; break;" % (i, qualify(v, val)))
                a("      default: nx = %s; break;" % qualify(v, vals[0]))
                a("      }")
            a("    }")
        a("    %s = nx;" % v)
        a("  }")
    a("}")
    a("")

    props = []
    for e in model.invarspec:
        props.append(("INVARSPEC", e))
    for e in model.ltlspec:
        s = e.strip()
        gm = re.fullmatch(r"G\s*\((.*)\)", s, re.S)
        if gm:
            props.append(("LTLSPEC G", gm.group(1).strip()))
        else:
            a("/* SKIPPED (not of the form G phi, so not a safety property):")
            a(" *   LTLSPEC %s */" % e)
    for e in model.ctlspec:
        a("/* SKIPPED (CTL has no ESBMC equivalent): SPEC %s */" % e)
    a("")

    a("int main(void)")
    a("{")
    for v, val in sorted(model.init.items()):
        a("  %s = %s;" % (v, qualify(v, val)))
    for v in free:
        a("  %s = (enum %s_e)0;" % (v, v))
    a("")
    a("  for (int _s = 0; _s < %d; _s++) {" % steps)
    a("    step_free();")
    a("")
    for name, expr in model.defines:
        try:
            a("    int %s = %s;" % (name, expr_to_c(expr, model)))
        except Unsupported as ex:
            a("    /* SKIPPED define %s: %s */" % (name, ex))
    a("")
    n = 0
    for kind, expr in props:
        if only_property and only_property not in expr:
            continue
        try:
            a("    assert(%s); /* %s */" % (expr_to_c(expr, model), kind))
            n += 1
        except Unsupported as ex:
            a("    /* SKIPPED %s %s: %s */" % (kind, expr, ex))
    a("")
    a("    step_fsm();")
    a("  }")
    a("  return 0;")
    a("}")
    a("")
    a("/* %d propert%s checked. */" % (n, "y" if n == 1 else "ies"))
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("-o", "--output")
    ap.add_argument("--steps", type=int, default=4,
                    help="transition steps to unroll (default 4)")
    ap.add_argument("--property",
                    help="only emit properties whose text contains this")
    args = ap.parse_args()

    src = Path(args.model)
    model = parse(src.read_text())
    print(f"parsed: {len(model.vars)} vars, {len(model.next)} state machines, "
          f"{len(model.defines)} defines, "
          f"{len(model.invarspec)} INVARSPEC, {len(model.ltlspec)} LTLSPEC, "
          f"{len(model.ctlspec)} SPEC, {len(model.constraints)} constraints "
          f"(dropped)", file=sys.stderr)

    out = generate(model, args.steps, src.name, args.property)
    if args.output:
        Path(args.output).write_text(out)
    else:
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
