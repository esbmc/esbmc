#!/usr/bin/env python3
"""Graphical PLCopen XML (tc6_0201) -> rung logic extractor for ESBMC-PLC."""

import re
import argparse
from pathlib import Path

import defusedxml.ElementTree as DefusedET

NS = "http://www.plcopen.org/xml/tc6_0201"
DefusedET.register_namespace = getattr(DefusedET, "register_namespace", lambda *a: None)


def tag(name):
    """Return qualified XML tag name."""
    return f"{{{NS}}}{name}"


def strip_xsi(content):
    """Remove xsi namespace attributes that confuse ElementTree."""
    content = re.sub(r' xmlns:xsi="[^"]*"', '', content)
    content = re.sub(r' xsi:schemaLocation="[^"]*"', '', content)
    return content


def parse_nodes(ld_elem):
    """Parse all LD elements into a node dict and build forward/backward edges."""
    nodes = {}
    for elem in ld_elem:
        lid = int(elem.get("localId", -1))
        negated = elem.get("negated", "false").lower() == "true"
        storage = elem.get("storage", "none").lower()
        var_el = elem.find(tag("variable"))
        var = var_el.text.strip() if var_el is not None and var_el.text else ""
        fed_by = [
            int(c.get("refLocalId"))
            for c in elem.iter(tag("connection"))
            if c.get("refLocalId")
        ]
        t = elem.tag
        if t == tag("leftPowerRail"):
            kind = "leftRail"
        elif t == tag("rightPowerRail"):
            kind = "rightRail"
        elif t == tag("contact"):
            kind = "contact"
        elif t == tag("coil"):
            kind = "coil"
        elif t == tag("block"):
            kind = "block"
            var = elem.get("typeName", "")
        else:
            continue
        nodes[lid] = {
            "kind": kind, "var": var,
            "negated": negated, "storage": storage,
            "fed_by": fed_by, "feeds": [],
        }

    for lid, node in nodes.items():
        for src in node["fed_by"]:
            if src in nodes:
                nodes[src]["feeds"].append(lid)

    return nodes


def find_paths(nodes, start, target, visited=None):
    """Find all paths from start to target in the node graph."""
    if visited is None:
        visited = set()
    if start == target:
        return [[target]]
    if start in visited:
        return []
    visited = visited | {start}
    paths = []
    for nxt in nodes.get(start, {}).get("feeds", []):
        for path in find_paths(nodes, nxt, target, visited):
            paths.append([start] + path)
    return paths


def path_to_expr(path, nodes):
    """Convert a path to a boolean AND expression of contacts."""
    terms = []
    for nid in path:
        node = nodes.get(nid, {})
        if node.get("kind") != "contact" or not node["var"]:
            continue
        terms.append(f"!{node['var']}" if node["negated"] else node["var"])
    return " && ".join(terms) if terms else "1"


def extract_rungs(nodes):
    """Extract rung logic from the node graph."""
    left_rails = [lid for lid, n in nodes.items() if n["kind"] == "leftRail"]
    coils = [lid for lid, n in nodes.items() if n["kind"] == "coil"]
    rungs = []
    for coil_id in coils:
        coil = nodes[coil_id]
        exprs = []
        for rail in left_rails:
            for path in find_paths(nodes, rail, coil_id):
                exprs.append(path_to_expr(path, nodes))
        if not exprs:
            continue
        final = " || ".join(f"({e})" for e in exprs) if len(exprs) > 1 else exprs[0]
        rungs.append({
            "var": coil["var"],
            "storage": coil["storage"],
            "negated": coil["negated"],
            "expr": final,
        })
    return rungs


def goto_ir(rungs):
    """Generate GOTO IR assignment statements from rungs."""
    lines = []
    for rung in rungs:
        var, expr, storage = rung["var"], rung["expr"], rung["storage"]
        if storage == "set":
            lines.append(f"  ASSIGN {var} = {var} || ({expr});  // SET")
        elif storage == "reset":
            lines.append(f"  ASSIGN {var} = {var} && !({expr});  // RESET")
        else:
            neg = "!" if rung["negated"] else ""
            lines.append(f"  ASSIGN {var} = {neg}({expr});")
    return lines


def convert(xml_path, verbose=False):
    """Convert graphical PLCopen XML to rung expressions."""
    content = strip_xsi(Path(xml_path).read_text(encoding="utf-8", errors="ignore"))
    try:
        root = DefusedET.fromstring(content)
    except DefusedET.ParseError as exc:
        print(f"ERROR: {exc}")
        return {}
    result = {}
    for pou in root.iter(tag("pou")):
        name = pou.get("name", "unknown")
        ld_body = pou.find(f".//{tag('LD')}")
        if ld_body is None:
            continue
        nodes = parse_nodes(ld_body)
        rungs = extract_rungs(nodes)
        result[name] = rungs
        if verbose:
            print(f"\nPOU: {name}  |  {len(nodes)} nodes  |  {len(rungs)} rungs")
            for rung in rungs:
                storage_str = f"[{rung['storage'].upper()}]" if rung["storage"] != "none" else ""
                print(f"  {rung['var']} {storage_str} = {rung['expr']}")
            print("  GOTO IR:")
            for line in goto_ir(rungs):
                print(line)
    return result


def main():
    """Entry point for the graphical LD converter."""
    parser = argparse.ArgumentParser(
        description="Graphical PLCopen XML -> rung extractor"
    )
    parser.add_argument("input")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    result = convert(args.input, verbose=True)
    total = sum(len(v) for v in result.values())
    print(f"\n✓ {len(result)} POU(s), {total} rung(s)")


if __name__ == "__main__":
    main()
