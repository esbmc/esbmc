"""Pass a bound method of a fixed instance on as a function (#8957).

Where `o` is bound once to an instance of a top-level class, `o.m` names the
same method of the same object wherever it is evaluated. A module-level
`def ESBMC_bound_o_m(x): return o.m(x)` is then the same callable, and unlike
a bound method it is a function value, which lists, returns and parameters
carry and calls through a value resolve.

Only a read whose value flows where function_value_types follows it is
rewritten: an argument of a function the module defines, a return of such a
function, or a value bound to a name, directly or as an element of a container
or a branch of a conditional expression. Any other read, such as a comparison,
an argument of a method or a value stored in an attribute, is refused as
before, and so is a method that takes defaults, `*args`, keyword-only
parameters or decorators, or that yields.
"""

import ast
import copy
from collections import Counter

from preprocessor.function_value_types import _Analysis

_KNOWN_RETURNS = {"int", "str", "float", "bool"}


def _plain_method(method):
    args = method.args
    return (isinstance(method, ast.FunctionDef) and not method.decorator_list and args.args and
            not (args.posonlyargs or args.vararg or args.kwarg or args.kwonlyargs or args.defaults)
            and not any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(method)))


def _instance(stmt, classes):
    """`(o, C)` for a statement `o = C(...)` of a class in `classes`, else None."""
    if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)):
        return None
    call = stmt.value
    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in classes:
        return stmt.targets[0].id, classes[call.func.id]
    return None


def _receivers(module, fixed_receivers):
    """For each `o` of `fixed_receivers`, the index of the statement that builds
    it and the plain methods of its class that `o.m` may name."""
    classes = {s.name: s for s in module.body if isinstance(s, ast.ClassDef)}
    found = {}
    for index, stmt in enumerate(module.body):
        built = _instance(stmt, classes)
        if built and built[0] in fixed_receivers:
            found[built[0]] = (index, {
                m.name: m
                for m in built[1].body if isinstance(m, ast.FunctionDef)
                and m.name in fixed_receivers[built[0]] and _plain_method(m)
            })
    return found


def _parents(module):
    return {id(c): n for n in ast.walk(module) for c in ast.iter_child_nodes(n)}


def _passed_on(node, parents, functions):
    """Whether the value of `node` flows where function_value_types follows it:
    to a name, to a parameter of one of `functions`, or out of one of them as a
    return, directly or as an element of a container or branch of a
    conditional expression."""
    parent = parents.get(id(node))
    if isinstance(parent, ast.keyword):
        parent = parents.get(id(parent))
    if isinstance(parent, ast.Call):
        return isinstance(parent.func, ast.Name) and parent.func.id in functions
    if isinstance(parent, ast.Return):
        return not _inside_class(parent, parents)
    if isinstance(parent, (ast.List, ast.Tuple, ast.Set)) or (isinstance(parent, ast.Dict)
                                                              and node in parent.values):
        return _held(parent, parents, functions)
    return isinstance(parent, ast.IfExp) and node is not parent.test and _held(
        parent, parents, functions)


def _held(container, parents, functions):
    parent = parents.get(id(container))
    if isinstance(parent, ast.Assign):
        return all(isinstance(t, ast.Name) for t in parent.targets)
    if isinstance(parent, ast.AnnAssign):
        return isinstance(parent.target, ast.Name)
    return _passed_on(container, parents, functions)


def _inside_class(node, parents):
    while id(node) in parents:
        node = parents[id(node)]
        if isinstance(node, ast.ClassDef):
            return True
    return False


def _functions(module):
    """The names bound once in `module`, by a function it defines at the top."""
    bound = Counter(n.id for n in ast.walk(module)
                    if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load))
    bound.update(n.name for n in ast.walk(module) if isinstance(n, (ast.FunctionDef, ast.ClassDef)))
    return {s.name for s in module.body if isinstance(s, ast.FunctionDef) and bound[s.name] == 1}


def _helper(receiver, method, returns):
    params = method.args.args[1:]
    name = f"ESBMC_bound_{receiver}_{method.name}"
    call = ast.Call(func=ast.Attribute(value=ast.Name(id=receiver, ctx=ast.Load()),
                                       attr=method.name,
                                       ctx=ast.Load()),
                    args=[ast.Name(id=p.arg, ctx=ast.Load()) for p in params],
                    keywords=[])
    args = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg=p.arg, annotation=copy.deepcopy(p.annotation)) for p in params],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[])
    helper = ast.FunctionDef(name=name,
                             args=args,
                             body=[ast.Return(value=call)],
                             decorator_list=[],
                             returns=returns,
                             type_params=[])
    for node in ast.walk(helper):
        ast.copy_location(node, method)
    return helper


def _declared_return(method, analysis):
    if method.returns is not None:
        return copy.deepcopy(method.returns)
    kind = analysis.def_kind(method)
    if kind == "None":
        return ast.Constant(value=None)
    return ast.Name(id=kind, ctx=ast.Load()) if kind in _KNOWN_RETURNS else None


def _read_sites(module, receivers):
    parents = _parents(module)
    functions = _functions(module)
    return [
        n for n in ast.walk(module) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load)
        and isinstance(n.value, ast.Name) and n.attr in receivers.get(n.value.id, (
            0, ()))[1] and _passed_on(n, parents, functions)
    ]


class _Reads(ast.NodeTransformer):
    """Replace the method reads named by `names`, keyed by node id."""

    def __init__(self, names):
        super().__init__()
        self.names = names

    def visit_Attribute(self, node):  # pylint: disable=invalid-name,missing-function-docstring
        name = self.names.get(id(node))
        if name is None:
            return self.generic_visit(node)
        return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)


def lower_method_values(module, fixed_receivers):
    """Replace each read of a method of a fixed instance whose value is passed
    on with the function that calls it, defined right after the statement that
    builds the instance, so a read that runs before it is a NameError as in
    CPython."""
    receivers = _receivers(module, fixed_receivers)
    sites = _read_sites(module, receivers)
    if not sites:
        return
    analysis = _Analysis(module, "")
    helpers = {}
    for site in sites:
        key = (site.value.id, site.attr)
        if key not in helpers:
            method = receivers[key[0]][1][key[1]]
            helpers[key] = _helper(key[0], method, _declared_return(method, analysis))
    _Reads({id(s): helpers[(s.value.id, s.attr)].name for s in sites}).visit(module)
    for receiver in sorted({o for o, _ in helpers}, key=lambda o: -receivers[o][0]):
        after = receivers[receiver][0] + 1
        module.body[after:after] = [h for (o, _), h in helpers.items() if o == receiver]
    ast.fix_missing_locations(module)
