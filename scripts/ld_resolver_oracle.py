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

Usage:
  scripts/ld_resolver_oracle.py [seeds] [depth] [sp|dag] [layers] [width]

ESBMC is taken from $ESBMC, else build/src/esbmc/esbmc. Exits non-zero if any
generated program does not verify, leaving the offending case in a
fail_seed<N>/ directory for inspection.
"""
import os
import random
import subprocess
import sys

ESBMC = os.environ.get("ESBMC", "build/src/esbmc/esbmc")


class Network:
    """Accumulates contact nodes as (localId, var, negated, predecessor ids)."""

    def __init__(self, rng, nvars):
        self.rng = rng
        self.nvars = nvars
        self.lid = 10
        self.nodes = []

    def new_lid(self):
        self.lid += 1
        return self.lid

    def contact(self, preds):
        var = "i%d" % self.rng.randrange(self.nvars)
        neg = self.rng.random() < 0.35
        lid = self.new_lid()
        self.nodes.append((lid, var, neg, list(preds)))
        return lid, ("!%s" % var) if neg else var

    def series_parallel(self, preds, depth):
        """Compose series (AND) and parallel (OR) groups; return (exits, expr)."""
        if depth <= 0 or self.rng.random() < 0.3:
            lid, expr = self.contact(preds)
            return [lid], expr
        if self.rng.random() < 0.5:
            exits_a, expr_a = self.series_parallel(preds, depth - 1)
            exits_b, expr_b = self.series_parallel(exits_a, depth - 1)
            return exits_b, "(%s && %s)" % (expr_a, expr_b)
        exits_a, expr_a = self.series_parallel(preds, depth - 1)
        exits_b, expr_b = self.series_parallel(preds, depth - 1)
        return exits_a + exits_b, "(%s || %s)" % (expr_a, expr_b)

    def layered_dag(self, layers, width):
        """Build a layered DAG; return (exit ids, power-flow formula)."""
        rail = "TRUE"
        prev = [(1, rail)]
        for _ in range(layers):
            cur = []
            for _ in range(width):
                chosen = self.rng.sample(
                    prev, self.rng.randrange(1, len(prev) + 1))
                lid, cond = self.contact([p[0] for p in chosen])
                feeds = [p[1] for p in chosen]
                if rail in feeds:
                    expr = cond  # energised straight from the rail
                else:
                    joined = feeds[0] if len(feeds) == 1 \
                        else "(" + " || ".join(feeds) + ")"
                    expr = "(%s && %s)" % (joined, cond)
                cur.append((lid, expr))
            prev = cur
        parts = [p[1] for p in prev]
        top = parts[0] if len(parts) == 1 else "(" + " || ".join(parts) + ")"
        return [p[0] for p in prev], top


def emit(seed, cfg):
    """Return (ld_xml, props_yaml, formula) for one generated network."""
    nvars = cfg["nvars"]
    net = Network(random.Random(seed), nvars)
    if cfg["mode"] == "dag":
        exits, expr = net.layered_dag(cfg["layers"], cfg["width"])
    else:
        exits, expr = net.series_parallel([1], cfg["depth"])
    coil = net.new_lid()

    out = [
        "<?xml version='1.0' encoding='utf-8'?>",
        '<project xmlns="http://www.plcopen.org/xml/tc6_0201">',
        '  <fileHeader companyName="ESBMC" productName="ld-oracle"'
        ' productVersion="1" creationDateTime="2026-07-30T00:00:00"/>',
        '  <contentHeader name="gen%d">' % seed,
        '    <coordinateInfo><ld><scaling x="10" y="10"/></ld>'
        '</coordinateInfo>',
        '  </contentHeader>',
        '  <types><dataTypes/><pous>',
        '      <pou name="gen" pouType="program">',
        '        <interface><localVars>',
    ]
    for v in range(nvars):
        out.append('          <variable name="i%d" address="%%IX0.%d">'
                   '<type><BOOL/></type></variable>' % (v, v))
    out += [
        '          <variable name="q" address="%QX0.0">'
        '<type><BOOL/></type></variable>',
        '        </localVars></interface>',
        '        <body><LD>',
        '          <leftPowerRail localId="1" width="10" height="40">',
        '            <position x="10" y="10"/>',
        '            <connectionPointOut><relPosition x="10" y="10"/>'
        '</connectionPointOut>',
        '          </leftPowerRail>',
    ]
    for (lid, var, neg, preds) in net.nodes:
        out.append('          <contact localId="%d" width="20" height="20"'
                   ' negated="%s">' % (lid, "true" if neg else "false"))
        out.append('            <position x="%d" y="10"/>' % (40 + 30 * lid))
        out.append('            <connectionPointIn>')
        out.append('              <relPosition x="0" y="10"/>')
        for p in preds:
            out.append('              <connection refLocalId="%d"/>' % p)
        out.append('            </connectionPointIn>')
        out.append('            <connectionPointOut>'
                   '<relPosition x="20" y="10"/></connectionPointOut>')
        out.append('            <variable>%s</variable>' % var)
        out.append('          </contact>')
    out.append('          <coil localId="%d" width="20" height="20"'
               ' negated="false">' % coil)
    out.append('            <position x="900" y="10"/>')
    out.append('            <connectionPointIn>')
    out.append('              <relPosition x="0" y="10"/>')
    for e in exits:
        out.append('              <connection refLocalId="%d"/>' % e)
    out.append('            </connectionPointIn>')
    out.append('            <connectionPointOut><relPosition x="20" y="10"/>'
               '</connectionPointOut>')
    out.append('            <variable>q</variable>')
    out.append('          </coil>')
    out += [
        '          <rightPowerRail localId="2" width="10" height="40">',
        '            <position x="950" y="10"/>',
        '            <connectionPointIn><relPosition x="0" y="10"/>'
        '<connection refLocalId="%d"/></connectionPointIn>' % coil,
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

    # Both implications, so an undriven coil (stuck false) fails the second one
    # rather than passing vacuously.
    props = ('properties:\n'
             '  - id: P1\n'
             '    kind: invariant\n'
             '    expression: "(!q || %s) && (q || !(%s))"\n'
             '    description: "coil equals its power flow"\n' % (expr, expr))
    return "\n".join(out) + "\n", props, expr


def verdict_of(output):
    """Extract ESBMC's verdict word, or an ERROR: line if there is none."""
    for v in ("VERIFICATION SUCCESSFUL", "VERIFICATION FAILED",
              "VERIFICATION UNKNOWN"):
        if v in output:
            return v.split()[-1]
    last = output.strip().splitlines()
    return "ERROR:" + (last[-1] if last else "<no output>")


def run_one(seed, tmpdir, cfg, timeout=60):
    """Generate one network, verify it, and return (verdict, formula, files)."""
    xml, props, expr = emit(seed, cfg)
    ld_path = os.path.join(tmpdir, "oracle.ld")
    props_path = os.path.join(tmpdir, "oracle.yaml")
    with open(ld_path, "w", encoding="utf-8") as f:
        f.write(xml)
    with open(props_path, "w", encoding="utf-8") as f:
        f.write(props)
    try:
        # ESBMC splits output across stdout and stderr; merge or the verdict
        # line is missed.
        proc = subprocess.run(
            [ESBMC, ld_path, "--ld-props", props_path,
             "--k-induction", "--max-k-step", "4"],
            capture_output=True, text=True, timeout=timeout, check=False)
        return verdict_of(proc.stdout + proc.stderr), expr, xml, props
    except subprocess.TimeoutExpired:
        return "TIMEOUT", expr, xml, props


def main():
    """Generate and verify a batch of networks; return 1 if any failed."""
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    mode = sys.argv[3] if len(sys.argv) > 3 else "sp"
    layers = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    width = int(sys.argv[5]) if len(sys.argv) > 5 else 2

    cfg = {"nvars": 3, "depth": depth, "mode": mode,
           "layers": layers, "width": width}
    tmpdir = os.environ.get("TMPDIR", "/tmp")
    tally = {}
    failures = []
    for seed in range(seeds):
        got, expr, xml, props = run_one(seed, tmpdir, cfg)
        tally[got] = tally.get(got, 0) + 1
        if got != "SUCCESSFUL":
            failures.append((seed, got, expr, xml, props))

    print("seeds=%d mode=%s depth=%d layers=%d width=%d -> %s"
          % (seeds, mode, depth, layers, width, tally))
    for (seed, got, expr, xml, props) in failures[:4]:
        print("\n--- seed %d: %s\n    formula: %s" % (seed, got, expr))
        outdir = "fail_seed%d" % seed
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "oracle.ld"), "w",
                  encoding="utf-8") as f:
            f.write(xml)
        with open(os.path.join(outdir, "oracle.yaml"), "w",
                  encoding="utf-8") as f:
            f.write(props)
        print("    saved to %s/" % outdir)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
