"""Unroll a loop over a fixed tuple of classes (#8765).

The converter cannot hold a class in a loop variable, so `for a in (str, int)`
read `a` as no class at all. Each element is a class name known statically, so
the loop and an `any`/`all` generator over it are expanded per class instead.
"""

# pylint: disable=invalid-name

import ast
import builtins
import copy
from collections import Counter

_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _binding_counts(module):
    counts = Counter()
    for n in ast.walk(module):
        if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load):
            counts[n.id] += 1
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            counts[n.name] += 1
        elif isinstance(n, ast.arg):
            counts[n.arg] += 1
        elif isinstance(n, ast.alias):
            counts[n.asname or n.name.split(".")[0]] += 1
        elif isinstance(n, ast.ExceptHandler) and n.name:
            counts[n.name] += 1
    return counts


def _loads(tree, name):
    return sum(
        isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)
        for n in ast.walk(tree))


def _jumps_out(body):
    """Whether `body` has a break or continue of its own loop."""
    stack = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Break, ast.Continue)):
            return True
        if not isinstance(node, (ast.For, ast.AsyncFor, ast.While, *_SCOPES)):
            stack.extend(ast.iter_child_nodes(node))
    return False


def _substitute(nodes, name, cls):
    out = copy.deepcopy(nodes)
    for tree in out:
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load):
                n.id = cls
    return out


class _Unroller(ast.NodeTransformer):

    def __init__(self, module):
        self.module = module
        self.counts = _binding_counts(module)
        known = {n for n, v in vars(builtins).items() if isinstance(v, type) and not self.counts[n]}
        known |= {
            s.name
            for s in module.body if isinstance(s, ast.ClassDef) and self.counts[s.name] == 1
        }
        self.known = known
        self.tuples = {
            s.targets[0].id: self._class_names(s.value)
            for s in module.body if isinstance(s, ast.Assign) and len(s.targets) == 1
            and isinstance(s.targets[0], ast.Name) and self.counts[s.targets[0].id] == 1
            and self._class_names(s.value)
        }

    def _class_names(self, node):
        if isinstance(node, ast.Name):
            return self.tuples.get(node.id) if hasattr(self, "tuples") else None
        if (isinstance(node, ast.Tuple) and node.elts
                and all(isinstance(e, ast.Name) and e.id in self.known for e in node.elts)):
            return [e.id for e in node.elts]
        return None

    def _loop_variable(self, target):
        """The loop variable's name when it is bound only by this loop."""
        if isinstance(target, ast.Name) and self.counts[target.id] == 1:
            return target.id
        return None

    def visit_For(self, node):
        """Expands the loop once per class when its body allows it."""
        self.generic_visit(node)
        name = self._loop_variable(node.target)
        classes = self._class_names(node.iter)
        if (not name or not classes or _jumps_out(node.body)
                or _loads(self.module, name) != sum(_loads(s, name) for s in node.body)):
            return node
        unrolled = [stmt for cls in classes for stmt in _substitute(node.body, name, cls)]
        return unrolled + node.orelse

    def visit_Call(self, node):
        """Turns any/all over the classes into an or/and of the element."""
        self.generic_visit(node)
        if not (isinstance(node.func, ast.Name) and node.func.id in ("any", "all")
                and not self.counts[node.func.id] and len(node.args) == 1 and not node.keywords
                and isinstance(node.args[0], ast.GeneratorExp)):
            return node
        gen = node.args[0]
        comp = gen.generators[0] if len(gen.generators) == 1 else None
        # A comprehension's variable is local to it, unlike a loop's.
        name = comp.target.id if comp and not comp.ifs and isinstance(comp.target,
                                                                      ast.Name) else None
        classes = self._class_names(comp.iter) if name else None
        is_bool = isinstance(gen.elt, ast.Compare) or (isinstance(gen.elt, ast.Call) and isinstance(
            gen.elt.func, ast.Name) and gen.elt.func.id == "isinstance")
        if not classes or not is_bool:
            return node
        values = [_substitute([gen.elt], name, cls)[0] for cls in classes]
        if len(values) == 1:
            return values[0]
        op = ast.Or() if node.func.id == "any" else ast.And()
        return ast.copy_location(ast.BoolOp(op=op, values=values), node)


def unroll_type_tuple_loops(module):
    """Unrolls every loop over a fixed tuple of classes in `module`."""
    _Unroller(module).visit(module)
    ast.fix_missing_locations(module)
