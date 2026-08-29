#!/usr/bin/env python3
"""Randomised correctness oracle for the ESBMC-PLC graphical LD resolver.

The resolver in src/ld-frontend/parser/plcopen_xml_parser.cpp turns a graphical
PLCopen LD body into rungs. Only two real graphical programs exist under
regression/ld/benchmarks/, which is thin cover for changes to it. This script
generates random networks, derives each one's power-flow formula independently
of the resolver, and asserts through ESBMC that the coil equals that formula.

The oracle is structural, not differential: it does not compare one resolver
against another (which would bless a shared bug) but against the ladder algebra
itself. Two network shapes are generated:

  sp   series-parallel, the shape a vendor tool draws: series = AND, parallel
       = OR.
  dag  layered, each node taking a random non-empty subset of the previous
       layer. Not series-parallel, so the formula comes from the per-node
       recurrence pf(n) = (OR over preds p of pf(p)) AND cond(n). A resolver
       that enumerates rail-to-coil paths computes the distributed form
       instead, so this mode is what checks distributivity.

Stateful constructs (edge contacts, feedback variables, function blocks,
set/reset coils) are out of scope by construction: the shadow update is emitted
before the scan-boundary assertion, so a previous-scan value is not observable
where properties are checked, and the property grammar has no temporal
operators. Those are pinned by the discriminating reachability tests under
regression/ld/ instead.

Usage:
  scripts/ld_resolver_oracle.py [seeds] [depth] [sp|dag] [layers] [width]

ESBMC is taken from $ESBMC, else build/src/esbmc/esbmc. Exits non-zero if any
generated program does not verify, leaving the offending case in a
fail_seed<N>/ directory for inspection.
"""
import os
import random
import subprocess  # nosec B404
import sys
import tempfile

ESBMC = os.environ.get("ESBMC", "build/src/esbmc/esbmc")
RAIL_LID = 1
RIGHT_RAIL_LID = 2


def _join_or(parts):
    """OR a list of formulas, parenthesising only when there is more than one."""
    if len(parts) == 1:
        return parts[0]
    joined = " || ".join(parts)
    return f"({joined})"


class Network:
    """Accumulates contact nodes as (localId, var, negated, predecessor ids)."""

    def __init__(self, rng, nvars):
        self.rng = rng
        self.nvars = nvars
        self.lid = 10
        self.nodes = []

    def new_lid(self):
        """Allocate the next unused PLCopen localId."""
        self.lid += 1
        return self.lid

    def contact(self, preds):
        """Append one contact fed by preds; return its (localId, condition)."""
        var = f"i{self.rng.randrange(self.nvars)}"
        negated = self.rng.random() < 0.35
        lid = self.new_lid()
        self.nodes.append((lid, var, negated, list(preds)))
        return lid, (f"!{var}" if negated else var)

    def series_parallel(self, preds, depth):
        """Compose series (AND) and parallel (OR) groups; return (exits, expr)."""
        if depth <= 0 or self.rng.random() < 0.3:
            lid, expr = self.contact(preds)
            return [lid], expr
        if self.rng.random() < 0.5:
            exits_a, expr_a = self.series_parallel(preds, depth - 1)
            exits_b, expr_b = self.series_parallel(exits_a, depth - 1)
            return exits_b, f"({expr_a} && {expr_b})"
        exits_a, expr_a = self.series_parallel(preds, depth - 1)
        exits_b, expr_b = self.series_parallel(preds, depth - 1)
        return exits_a + exits_b, f"({expr_a} || {expr_b})"

    def layered_dag(self, layers, width):
        """Build a layered DAG; return (exit ids, power-flow formula)."""
        rail = "TRUE"
        prev = [(RAIL_LID, rail)]
        for _ in range(layers):
            cur = []
            for _ in range(width):
                chosen = self.rng.sample(prev,
                                         self.rng.randrange(1, len(prev) + 1))
                lid, cond = self.contact([p[0] for p in chosen])
                feeds = [p[1] for p in chosen]
                if rail in feeds:
                    expr = cond  # energised straight from the rail
                else:
                    expr = f"({_join_or(feeds)} && {cond})"
                cur.append((lid, expr))
            prev = cur
        return [p[0] for p in prev], _join_or([p[1] for p in prev])


def _interface_xml(seed, nvars):
    """Project header through the end of the variable declarations."""
    lines = [
        "<?xml version='1.0' encoding='utf-8'?>",
        '<project xmlns="http://www.plcopen.org/xml/tc6_0201">',
        '  <fileHeader companyName="ESBMC" productName="ld-oracle"'
        ' productVersion="1" creationDateTime="2026-07-30T00:00:00"/>',
        f'  <contentHeader name="gen{seed}">',
        '    <coordinateInfo><ld><scaling x="10" y="10"/></ld>'
        '</coordinateInfo>',
        '  </contentHeader>',
        '  <types><dataTypes/><pous>',
        '      <pou name="gen" pouType="program">',
        '        <interface><localVars>',
    ]
    for var in range(nvars):
        lines.append(f'          <variable name="i{var}" address="%IX0.{var}">'
                     '<type><BOOL/></type></variable>')
    lines += [
        '          <variable name="q" address="%QX0.0">'
        '<type><BOOL/></type></variable>',
        '        </localVars></interface>',
        '        <body><LD>',
        f'          <leftPowerRail localId="{RAIL_LID}" width="10" '
        'height="40">',
        '            <position x="10" y="10"/>',
        '            <connectionPointOut><relPosition x="10" y="10"/>'
        '</connectionPointOut>',
        '          </leftPowerRail>',
    ]
    return lines


def _contact_xml(node):
    """One <contact> element, wired to each of its predecessors."""
    lid, var, negated, preds = node
    negated_attr = "true" if negated else "false"
    lines = [
        f'          <contact localId="{lid}" width="20" height="20"'
        f' negated="{negated_attr}">',
        f'            <position x="{40 + 30 * lid}" y="10"/>',
        '            <connectionPointIn>',
        '              <relPosition x="0" y="10"/>',
    ]
    for pred in preds:
        lines.append(f'              <connection refLocalId="{pred}"/>')
    lines += [
        '            </connectionPointIn>',
        '            <connectionPointOut><relPosition x="20" y="10"/>'
        '</connectionPointOut>',
        f'            <variable>{var}</variable>',
        '          </contact>',
    ]
    return lines


def _sink_xml(coil, exits):
    """The output coil, its incoming connections, and the closing elements."""
    lines = [
        f'          <coil localId="{coil}" width="20" height="20"'
        ' negated="false">',
        '            <position x="900" y="10"/>',
        '            <connectionPointIn>',
        '              <relPosition x="0" y="10"/>',
    ]
    for exit_lid in exits:
        lines.append(f'              <connection refLocalId="{exit_lid}"/>')
    lines += [
        '            </connectionPointIn>',
        '            <connectionPointOut><relPosition x="20" y="10"/>'
        '</connectionPointOut>',
        '            <variable>q</variable>',
        '          </coil>',
        f'          <rightPowerRail localId="{RIGHT_RAIL_LID}" width="10"'
        ' height="40">',
        '            <position x="950" y="10"/>',
        '            <connectionPointIn><relPosition x="0" y="10"/>'
        f'<connection refLocalId="{coil}"/></connectionPointIn>',
        '          </rightPowerRail>',
        '        </LD></body>',
        '      </pou>',
        '  </pous></types>',
        '  <instances><configurations><configuration name="c">'
        '<resource name="r">',
        '    <task name="main" interval="T#10ms" priority="0">'
        '<pouInstance name="p" typeName="gen"/></task>',
        '  </resource></configuration></configurations></instances>',
        '</project>',
    ]
    return lines


def emit(seed, cfg):
    """Return (ld_xml, props_yaml, formula) for one generated network."""
    net = Network(random.Random(seed), cfg["nvars"])  # nosec B311
    if cfg["mode"] == "dag":
        exits, expr = net.layered_dag(cfg["layers"], cfg["width"])
    else:
        exits, expr = net.series_parallel([RAIL_LID], cfg["depth"])

    lines = _interface_xml(seed, cfg["nvars"])
    for node in net.nodes:
        lines += _contact_xml(node)
    lines += _sink_xml(net.new_lid(), exits)

    # Both implications, so an undriven coil (stuck false) fails the second one
    # rather than passing vacuously.
    props = ("properties:\n"
             "  - id: P1\n"
             "    kind: invariant\n"
             f'    expression: "(!q || {expr}) && (q || !({expr}))"\n'
             '    description: "coil equals its power flow"\n')
    return "\n".join(lines) + "\n", props, expr


def verdict_of(output):
    """Extract ESBMC's verdict word, or an ERROR: line if there is none."""
    for verdict in ("VERIFICATION SUCCESSFUL", "VERIFICATION FAILED",
                    "VERIFICATION UNKNOWN"):
        if verdict in output:
            return verdict.split()[-1]
    tail = output.strip().splitlines()
    return "ERROR:" + (tail[-1] if tail else "<no output>")


def run_one(seed, tmpdir, cfg, timeout=60):
    """Generate one network, verify it, and return (verdict, formula, files)."""
    xml, props, expr = emit(seed, cfg)
    ld_path = os.path.join(tmpdir, "oracle.ld")
    props_path = os.path.join(tmpdir, "oracle.yaml")
    with open(ld_path, "w", encoding="utf-8") as handle:
        handle.write(xml)
    with open(props_path, "w", encoding="utf-8") as handle:
        handle.write(props)
    try:
        # ESBMC splits output across stdout and stderr; merge or the verdict
        # line is missed.
        proc = subprocess.run(  # nosec B603
            [ESBMC, ld_path, "--ld-props", props_path, "--k-induction",
             "--max-k-step", "4"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False)
        return verdict_of(proc.stdout + proc.stderr), expr, xml, props
    except subprocess.TimeoutExpired:
        return "TIMEOUT", expr, xml, props


def _save_failure(seed, xml, props):
    """Write a failing case to fail_seed<N>/ and return the directory name."""
    outdir = f"fail_seed{seed}"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "oracle.ld"), "w",
              encoding="utf-8") as handle:
        handle.write(xml)
    with open(os.path.join(outdir, "oracle.yaml"), "w",
              encoding="utf-8") as handle:
        handle.write(props)
    return outdir


def main():
    """Generate and verify a batch of networks; return 1 if any failed."""
    argv = sys.argv[1:]
    seeds = int(argv[0]) if argv else 30
    cfg = {
        "nvars": 3,
        "depth": int(argv[1]) if len(argv) > 1 else 3,
        "mode": argv[2] if len(argv) > 2 else "sp",
        "layers": int(argv[3]) if len(argv) > 3 else 3,
        "width": int(argv[4]) if len(argv) > 4 else 2,
    }
    tally = {}
    failures = []
    # A private directory per run: the generated files reuse one name, so two
    # concurrent runs sharing $TMPDIR would overwrite each other's input.
    with tempfile.TemporaryDirectory(prefix="ld-oracle-") as tmpdir:
        for seed in range(seeds):
            got, expr, xml, props = run_one(seed, tmpdir, cfg)
            tally[got] = tally.get(got, 0) + 1
            if got != "SUCCESSFUL":
                failures.append((seed, got, expr, xml, props))

    print(f"seeds={seeds} mode={cfg['mode']} depth={cfg['depth']} "
          f"layers={cfg['layers']} width={cfg['width']} -> {tally}")
    for (seed, got, expr, xml, props) in failures[:4]:
        print(f"\n--- seed {seed}: {got}\n    formula: {expr}")
        print(f"    saved to {_save_failure(seed, xml, props)}/")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
