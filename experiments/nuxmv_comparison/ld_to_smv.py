#!/usr/bin/env python3
"""
ld_to_smv.py  —  PLCopen XML Ladder Diagram → NuXmv SMV transpiler

Scan-cycle model (matches ESBMC-PLC's GOTO IR / C model):
  - Inputs are state VARs resampled nondeterministically each scan.
    This ensures the SAME input value is used for both rung computation
    and property checking within one scan step (avoiding the IVAR
    semantic mismatch where inputs can differ between computation and check).
  - Rungs execute in document order with immediate-update semantics:
    each rung's contact reads the value produced by the most recent
    earlier rung that drove the same variable (or the previous-scan
    state if no earlier rung drove it).  This is implemented via a
    DEFINE chain: define __scanN_var := condition; later rungs
    reference __scanN_var instead of var.
  - Timer (TON/TOF) elapsed time is persistent state (VAR + ASSIGN next()).
  - All combinational outputs are SMV DEFINE expressions.
  - Safety properties use INVARSPEC (BDD reachability, same as PLCverif's
    nuXmv mode).

Usage:
  python3 ld_to_smv.py  program.ld  props.yaml  [--out program.smv]
"""

import argparse
import re
import sys
from collections import defaultdict

import defusedxml.ElementTree as XmlET
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tag(elem):
    return re.sub(r'\{[^}]*\}', '', elem.tag)


IEC_BOOL_TYPES = {'BOOL'}
IEC_INT_TYPES  = {'INT', 'DINT', 'UINT', 'SINT', 'USINT', 'WORD', 'BYTE', 'TIME', 'DWORD'}

SMV_BOOL = 'boolean'
SMV_INT  = '0 .. 32767'

def smv_type(iec):
    iec = iec.upper()
    if iec in IEC_BOOL_TYPES:
        return SMV_BOOL
    return SMV_INT

def is_bool_type(iec):
    return iec.upper() in IEC_BOOL_TYPES

def smv_init(iec):
    return 'FALSE' if is_bool_type(iec) else '0'


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
class LDParser:
    def __init__(self, path):
        tree = XmlET.parse(path)
        root = tree.getroot()
        self.inputs  = {}
        self.outputs = {}
        self.locals  = {}
        self.rungs   = []
        self._parse(root)

    def _parse(self, root):
        for pou in root.iter():
            if _tag(pou) != 'pou':
                continue
            self._collect_vars(pou, 'inputVars',  self.inputs)
            self._collect_vars(pou, 'outputVars', self.outputs)
            self._collect_vars(pou, 'localVars',  self.locals)
            for ld in pou.iter():
                if _tag(ld) == 'LD':
                    self._parse_ld(ld)
            break

    def _collect_vars(self, pou, section, target):
        for sec in pou.iter():
            if _tag(sec) != section:
                continue
            for v in sec:
                if _tag(v) != 'variable':
                    continue
                name = v.get('name')
                iec = 'BOOL'
                for t in v.iter():
                    tname = _tag(t)
                    if tname in IEC_BOOL_TYPES | IEC_INT_TYPES:
                        iec = tname
                        break
                target[name] = iec

    def _parse_ld(self, ld_elem):
        for child in ld_elem:
            if _tag(child) != 'rung':
                continue
            r = _Rung()
            for elem in child:
                tag = _tag(elem)
                if tag == 'contact':
                    var = elem.get('variable', '')
                    neg = elem.get('negated', '') in ('negated', 'true', '1', 'True')
                    r.contacts.append((var, neg))
                elif tag == 'coil':
                    var = elem.get('variable', '')
                    st  = elem.get('storage', 'none')
                    if st in ('set', 'S'):
                        st = 'set'
                    elif st in ('reset', 'R'):
                        st = 'reset'
                    else:
                        st = 'normal'
                    r.coils.append((var, st))
                elif tag == 'block':
                    tname = elem.get('typeName', '')
                    iname = elem.get('instanceName', tname)
                    params = {}
                    for p in elem:
                        if _tag(p) == 'variable':
                            fp = p.get('formalParameter', '')
                            params[fp] = p.text or p.get('name', '')
                    r.blocks.append((tname.upper(), iname, params))
            self.rungs.append(r)


class _Rung:
    def __init__(self):
        self.contacts = []
        self.coils    = []
        self.blocks   = []


# ---------------------------------------------------------------------------
# SMV Generator
# ---------------------------------------------------------------------------
class SMVGen:
    def __init__(self, parser, props):
        self.p     = parser
        self.props = props
        self.all_vars = {**parser.inputs, **parser.outputs, **parser.locals}

        # Identify which vars are driven by coils (updated this scan)
        self.coil_driven = set()
        for r in parser.rungs:
            for (var, _) in r.coils:
                self.coil_driven.add(var)
            for (_, iname, params) in r.blocks:
                for fp in ('Q', 'ET'):
                    v = params.get(fp, '')
                    if v:
                        self.coil_driven.add(v)

        # Collect timer instances
        self.timers = {}
        for r in parser.rungs:
            for (tname, iname, params) in r.blocks:
                if tname in ('TON', 'TOF', 'TP'):
                    self.timers[iname] = {
                        'type': tname,
                        'et_var': params.get('ET', f'__{iname}_et'),
                        'q_var':  params.get('Q',  f'__{iname}_q'),
                        'pt_var': params.get('PT', '0'),
                        'in_var': params.get('IN', 'FALSE'),
                    }

    def generate(self):
        state_vars, defines, next_assigns, invarspecs = self._build_model()

        lines = ['MODULE main', '']
        lines.append('VAR')
        for name, typ in sorted(state_vars.items()):
            lines.append(f'  {name} : {typ};')
        lines.append('')

        if defines:
            lines.append('DEFINE')
            for name, expr in defines:
                # Handle multi-line case expressions
                expr_lines = expr.split('\n')
                lines.append(f'  {name} :=')
                for el in expr_lines:
                    lines.append(f'    {el}')
                lines[-1] += ';'
            lines.append('')

        lines.append('ASSIGN')
        for stmt in next_assigns:
            lines.append(f'  {stmt}')
        lines.append('')

        for comment, spec in invarspecs:
            lines.append(f'-- {comment}')
            lines.append(spec)
            lines.append('')

        return '\n'.join(lines)

    def _collect_state_vars(self):
        state_vars = {}
        for name, iec in self.p.inputs.items():
            state_vars[name] = smv_type(iec)
        for name, iec in self.p.locals.items():
            if name not in self.coil_driven:
                state_vars[name] = smv_type(iec)
        for name, iec in self.p.outputs.items():
            if name not in self.coil_driven:
                state_vars[name] = smv_type(iec)
        for iname, t in self.timers.items():
            et = t['et_var']
            if et not in state_vars:
                state_vars[et] = SMV_INT
        return state_vars

    def _process_blocks(self, rung, latest, defines, coil_normal_define, timer_et_nexts):
        for tname, iname, params in rung.blocks:
            in_raw = params.get('IN', 'FALSE')
            pt_raw = params.get('PT', '0')
            in_v = latest.get(in_raw, in_raw)
            pt_v = latest.get(pt_raw, pt_raw)
            et_v = params.get('ET', f'__{iname}_et')
            q_p  = params.get('Q',  f'__{iname}_q')
            if tname == 'TON':
                et_expr = (f'case\n  {in_v} & {et_v} < 32767 : {et_v} + 1;\n'
                           f'  !({in_v}) : 0;\n  TRUE : {et_v};\nesac')
                timer_et_nexts[et_v] = et_expr
                q_dname = f'__ton_{iname}_q_v'
                defines.append((q_dname, f'{et_v} >= {pt_v}'))
                latest[q_p] = q_dname
                if q_p in self.all_vars:
                    coil_normal_define[q_p] = q_dname
            elif tname == 'TOF':
                et_expr = (f'case\n  !({in_v}) & {et_v} < 32767 : {et_v} + 1;\n'
                           f'  {in_v} : 0;\n  TRUE : {et_v};\nesac')
                timer_et_nexts[et_v] = et_expr
                q_dname = f'__tof_{iname}_q_v'
                defines.append((q_dname, f'{in_v} | ({et_v} < {pt_v})'))
                latest[q_p] = q_dname
                if q_p in self.all_vars:
                    coil_normal_define[q_p] = q_dname

    def _merge_sr_coils(self, latest, defines, coil_normal_define, coil_set_conds, coil_reset_conds):
        for var in set(coil_set_conds) | set(coil_reset_conds):
            sc   = coil_set_conds.get(var, [])
            rc   = coil_reset_conds.get(var, [])
            se   = ' | '.join(f'({c})' for c in sc) or 'FALSE'
            re_  = ' | '.join(f'({c})' for c in rc) or 'FALSE'
            dname = f'__sr_{var}_v'
            prev  = latest.get(var, var)
            expr  = (f'case\n  {se} : TRUE;\n  {re_} : FALSE;\n  TRUE : {prev};\nesac')
            defines.append((dname, expr))
            latest[var] = dname
            coil_normal_define[var] = dname

    def _build_define_chain(self):
        latest = {v: v for v in self.all_vars}
        for iname, t in self.timers.items():
            latest[t['et_var']] = t['et_var']
        defines = []
        coil_normal_define = {}
        coil_set_conds = defaultdict(list)
        coil_reset_conds = defaultdict(list)
        timer_et_nexts = {}
        for ridx, rung in enumerate(self.p.rungs):
            cond_parts = []
            for var, neg in rung.contacts:
                v = latest.get(var, var)
                cond_parts.append(f'!({v})' if neg else v)
            cond = ' & '.join(cond_parts) if cond_parts else 'TRUE'
            for var, storage in rung.coils:
                dname = f'__r{ridx}_{var}'
                if storage == 'normal':
                    defines.append((dname, cond))
                    latest[var] = dname
                    coil_normal_define[var] = dname
                elif storage == 'set':
                    coil_set_conds[var].append(cond)
                elif storage == 'reset':
                    coil_reset_conds[var].append(cond)
            self._process_blocks(rung, latest, defines, coil_normal_define, timer_et_nexts)
        self._merge_sr_coils(latest, defines, coil_normal_define, coil_set_conds, coil_reset_conds)
        return latest, defines, coil_normal_define, timer_et_nexts

    def _build_next_assigns(self, coil_normal_define, timer_et_nexts, state_vars):
        p = self.p
        next_assigns = []
        for name, iec in sorted(p.inputs.items()):
            next_assigns.append(f'init({name}) := {smv_init(iec)};')
        for name, iec in sorted(p.inputs.items()):
            if is_bool_type(iec):
                next_assigns.append(f'next({name}) := {{TRUE, FALSE}};')
            else:
                next_assigns.append(f'next({name}) := 0 .. 32767;')
        for name, iec in sorted(p.locals.items()):
            if name not in self.coil_driven:
                next_assigns.append(f'init({name}) := {smv_init(iec)};')
                next_assigns.append(f'next({name}) := {name};')
        for name, iec in sorted(p.outputs.items()):
            if name not in self.coil_driven:
                next_assigns.append(f'init({name}) := {smv_init(iec)};')
                next_assigns.append(f'next({name}) := {name};')
        for et_var, et_expr in sorted(timer_et_nexts.items()):
            next_assigns.append(f'init({et_var}) := 0;')
            et_lines = et_expr.split('\n')
            next_assigns.append(f'next({et_var}) :=')
            for line in et_lines:
                next_assigns.append(f'  {line}')
            next_assigns[-1] += ';'
        for var in sorted(self.coil_driven):
            if var in coil_normal_define:
                dname = coil_normal_define[var]
                iec = self.all_vars.get(var, 'BOOL')
                if var not in state_vars:
                    state_vars[var] = smv_type(iec)
                next_assigns.append(f'init({var}) := {smv_init(iec)};')
                next_assigns.append(f'next({var}) := {dname};')
        return next_assigns

    def _build_invarspecs(self, latest):
        invarspecs = []
        for prop in self.props:
            kind    = prop.get('kind', '')
            pid     = prop.get('id', '?')
            desc    = prop.get('description', '')
            comment = f'{pid}: {desc}'
            if kind == 'mutual_exclusion':
                vs   = prop['variables']
                v0   = latest.get(vs[0], vs[0])
                v1   = latest.get(vs[1], vs[1])
                spec = f'INVARSPEC !({v0} & {v1});'
            elif kind == 'invariant':
                expr = self._xlat(prop['expression'], latest)
                spec = f'INVARSPEC {expr};'
            elif kind == 'absence':
                expr = self._xlat(prop['expression'], latest)
                spec = f'INVARSPEC !({expr});'
            else:
                comment += f' [{kind} — skipped in NuXmv comparison]'
                spec = ''
            if spec:
                invarspecs.append((comment, spec))
        return invarspecs

    def _build_model(self):
        state_vars = self._collect_state_vars()
        latest, defines, coil_normal_define, timer_et_nexts = self._build_define_chain()
        next_assigns = self._build_next_assigns(coil_normal_define, timer_et_nexts, state_vars)
        invarspecs = self._build_invarspecs(latest)
        return state_vars, defines, next_assigns, invarspecs

    def _xlat(self, expr, latest):
        expr = expr.replace('&&', '&').replace('||', '|')

        def repl(m):
            v = m.group(0)
            return latest.get(v, v)

        return re.sub(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', repl, expr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ld')
    ap.add_argument('props')
    ap.add_argument('--out')
    args = ap.parse_args()

    parser = LDParser(args.ld)
    with open(args.props, encoding='utf-8') as f:
        data = yaml.safe_load(f)
    props = data.get('properties', [])

    gen = SMVGen(parser, props)
    smv = gen.generate()

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(smv)
        print(f'[ld_to_smv] Written: {args.out}', file=sys.stderr)
    else:
        print(smv)


if __name__ == '__main__':
    main()
