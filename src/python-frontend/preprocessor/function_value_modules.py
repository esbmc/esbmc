"""Lift the functions read as attributes of an imported module to module-level functions (#8970).

`f = sub.func if c else pkg.func` binds `f` to a function of one of two modules. The dispatch
pass follows module-level functions by name, and the two may share one. When `sub` is bound
only by its `import`, is only read as the receiver of an attribute, and `func` is the one plain
def of that name in `sub`'s source, the value `sub.func` is the function
`def _esbmc_mf_sub_func(x): return sub.func(x)`. This pass makes that lift, on a copy of the
module, for every such value that is not called on the spot.
"""

import ast

from preprocessor.function_value_methods import (_DYNAMIC, KIND_ATTRIBUTE, _binding_count,
                                                 _receiver_uses_are_attributes,
                                                 _stores_a_method_name)
from preprocessor.function_value_types import _Analysis

LIFTED = "_esbmc_mf_"
# The `__name__` of the function a lifted def stands for.
DEFINED_NAME = "esbmc_defined_name"


def _module_imports(module):
    """Each name a top-level `import` binds to a whole module, with that statement."""
    return {
        (alias.asname or alias.name): (stmt, alias.name)
        for stmt in module.body if isinstance(stmt, ast.Import) for alias in stmt.names
        if alias.asname or "." not in alias.name
    }


def _sole_function(analysis, dotted, name):
    """The one plain def `name` in the local module `dotted`, or None."""
    path = analysis.local_module(dotted)
    if path is None:
        return None
    fn = analysis.module_def(path, name) or analysis.aliased_def(path, name)
    if fn is None or fn.decorator_list or _binding_count(analysis.loaded[path], name) != 1:
        return None
    args = fn.args
    plain = not (args.vararg or args.kwarg or args.kwonlyargs or args.defaults or args.posonlyargs)
    return fn if plain else None


def _wrapper(receiver, name, fn, anchor):
    params = [a.arg for a in fn.args.args]
    call = ast.Call(ast.Attribute(ast.Name(receiver, ast.Load()), name, ast.Load()),
                    [ast.Name(p, ast.Load()) for p in params], [])
    wrapper = ast.FunctionDef(name=f"{LIFTED}{receiver}_{name}",
                              args=ast.arguments([], [ast.arg(p) for p in params], None, [], [],
                                                 None, []),
                              body=[ast.Return(call)],
                              decorator_list=[],
                              returns=None,
                              type_comment=None,
                              type_params=[])
    setattr(wrapper, KIND_ATTRIBUTE, f"module:{receiver}.{name}")
    setattr(wrapper, DEFINED_NAME, fn.name)
    return ast.copy_location(wrapper, anchor)


class _ReadLifted(ast.NodeTransformer):
    """Read each of the given `receiver.name` values as its lifted function."""

    def __init__(self, receiver, values):
        self.receiver = receiver
        self.values = values

    def visit_Attribute(self, node):
        if id(node) not in self.values:
            return self.generic_visit(node)
        name = ast.Name(f"{LIFTED}{self.receiver}_{node.attr}", ast.Load())
        return ast.copy_location(name, node)


def _values(module, receiver):
    """The reads `receiver.name` that are not the callee of a call."""
    callees = {id(n.func) for n in ast.walk(module) if isinstance(n, ast.Call)}
    return [
        n for n in ast.walk(module)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == receiver
        and isinstance(n.ctx, ast.Load) and id(n) not in callees
    ]


def _liftable(module, analysis, receiver, dotted, values):
    """The functions `values` read from `receiver`, when each can be lifted, else None."""
    if not values or _binding_count(module, receiver) != 1:
        return None
    attributes = {v.attr for v in values}
    if not _receiver_uses_are_attributes(module, receiver) or _stores_a_method_name(
            module, attributes):
        return None
    functions = {v.attr: _sole_function(analysis, dotted, v.attr) for v in values}
    if None in functions.values() or any(p.arg == receiver for fn in functions.values()
                                         for p in fn.args.args):
        return None
    return functions


def lift_module_functions(module, path):
    """Lift in place the functions read as values of imported modules; return whether any was."""
    if {n.id for n in ast.walk(module) if isinstance(n, ast.Name)} & _DYNAMIC:
        return False
    analysis = _Analysis(module, path)
    lifts = []
    for receiver, (stmt, dotted) in _module_imports(module).items():
        values = _values(module, receiver)
        functions = _liftable(module, analysis, receiver, dotted, values)
        if functions is not None:
            lifts.append((receiver, stmt, values, functions))
    # An alias reads a def under a second name, and the two lifts would compare unequal.
    reached = [id(fn) for *_, functions in lifts for fn in functions.values()]
    if len(reached) != len(set(reached)):
        return False
    for receiver, stmt, values, functions in lifts:
        _ReadLifted(receiver, {id(v) for v in values}).visit(module)
        position = module.body.index(stmt) + 1
        module.body[position:position] = [
            _wrapper(receiver, name, fn, stmt) for name, fn in functions.items()
        ]
        ast.fix_missing_locations(module)
    return bool(lifts)
