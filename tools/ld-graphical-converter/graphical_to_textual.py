#!/usr/bin/env python3
import sys, re, argparse
from pathlib import Path
from xml.etree import ElementTree as ET

NS = "http://www.plcopen.org/xml/tc6_0201"
ET.register_namespace("", NS)
ET.register_namespace("xhtml", "http://www.w3.org/1999/xhtml")

def tag(n): return f"{{{NS}}}{n}"

def strip_xsi(c):
    c = re.sub(r' xmlns:xsi="[^"]*"', '', c)
    c = re.sub(r' xsi:schemaLocation="[^"]*"', '', c)
    return c

def parse_nodes(ld_elem):
    nodes = {}
    for elem in ld_elem:
        lid     = int(elem.get("localId", -1))
        negated = elem.get("negated","false").lower() == "true"
        storage = elem.get("storage","none").lower()
        ve      = elem.find(tag("variable"))
        var     = ve.text.strip() if ve is not None and ve.text else ""
        fed_by  = [int(c.get("refLocalId")) for c in elem.iter(tag("connection")) if c.get("refLocalId")]
        t = elem.tag
        if   t == tag("leftPowerRail"):  kind = "leftRail"
        elif t == tag("rightPowerRail"): kind = "rightRail"
        elif t == tag("contact"):        kind = "contact"
        elif t == tag("coil"):           kind = "coil"
        elif t == tag("block"):          kind = "block"; var = elem.get("typeName","")
        else: continue
        nodes[lid] = {"kind":kind,"var":var,"negated":negated,"storage":storage,"fed_by":fed_by,"feeds":[]}
    for lid, n in nodes.items():
        for src in n["fed_by"]:
            if src in nodes: nodes[src]["feeds"].append(lid)
    return nodes

def find_paths(nodes, start, target, visited=None):
    if visited is None: visited = set()
    if start == target: return [[target]]
    if start in visited: return []
    visited = visited | {start}
    paths = []
    for nxt in nodes.get(start, {}).get("feeds", []):
        for p in find_paths(nodes, nxt, target, visited):
            paths.append([start] + p)
    return paths

def path_to_expr(path, nodes):
    terms = []
    for nid in path:
        n = nodes.get(nid, {})
        if n.get("kind") != "contact" or not n["var"]: continue
        terms.append(f"!{n['var']}" if n["negated"] else n["var"])
    return " && ".join(terms) if terms else "1"

def extract_rungs(nodes):
    left_rails = [lid for lid,n in nodes.items() if n["kind"]=="leftRail"]
    coils      = [lid for lid,n in nodes.items() if n["kind"]=="coil"]
    rungs = []
    for cid in coils:
        c = nodes[cid]
        exprs = []
        for rail in left_rails:
            for path in find_paths(nodes, rail, cid):
                exprs.append(path_to_expr(path, nodes))
        if not exprs: continue
        final = " || ".join(f"({e})" for e in exprs) if len(exprs)>1 else exprs[0]
        rungs.append({"var":c["var"],"storage":c["storage"],"negated":c["negated"],"expr":final})
    return rungs

def goto_ir(rungs):
    lines = []
    for r in rungs:
        v,e,s = r["var"],r["expr"],r["storage"]
        if   s == "set":   lines.append(f"  ASSIGN {v} = {v} || ({e});  // SET")
        elif s == "reset": lines.append(f"  ASSIGN {v} = {v} && !({e});  // RESET")
        else:
            neg = "!" if r["negated"] else ""
            lines.append(f"  ASSIGN {v} = {neg}({e});")
    return lines

def convert(xml_path, verbose=False):
    content = strip_xsi(Path(xml_path).read_text(errors="ignore"))
    try:    root = ET.fromstring(content)
    except ET.ParseError as e: print(f"ERROR: {e}"); return {}
    result = {}
    for pou in root.iter(tag("pou")):
        name = pou.get("name","unknown")
        ld   = pou.find(f".//{tag('LD')}")
        if ld is None: continue
        nodes = parse_nodes(ld)
        rungs = extract_rungs(nodes)
        result[name] = rungs
        if verbose:
            print(f"\nPOU: {name}  |  {len(nodes)} nodes  |  {len(rungs)} rungs")
            for r in rungs:
                st = f"[{r['storage'].upper()}]" if r["storage"] != "none" else ""
                print(f"  {r['var']} {st} = {r['expr']}")
            print("  GOTO IR:")
            for line in goto_ir(rungs): print(line)
    return result

def main():
    ap = argparse.ArgumentParser(description="Graphical PLCopen XML -> rung extractor")
    ap.add_argument("input")
    ap.add_argument("-v","--verbose",action="store_true")
    args = ap.parse_args()
    r = convert(args.input, verbose=True)
    total = sum(len(v) for v in r.values())
    print(f"\n✓ {len(r)} POU(s), {total} rung(s)")

if __name__ == "__main__": main()
