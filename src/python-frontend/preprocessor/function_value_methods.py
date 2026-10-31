"""Lift the bound methods a method returns to module-level functions (#8957).

`def pick(self, flag): return self.f if flag else self.g`, called as `a.pick(flag)` on the one
module-level instance `a`, returns a bound method of `a`. The dispatch pass cannot follow a
bound method, but it follows module-level functions. When the receiver is statically `a`, the
bound method `a.f` is the function `lambda: a.f()`, and the call `a.pick(flag)` is the call of
`pick`'s body with `self` read as `a`. This pass makes both lifts, on a copy of the module, and
leaves the class as it was. Anything it cannot prove keeps the module unchanged.
"""

import ast

from preprocessor.function_value_types import _own_nodes

BOUND = "_esbmc_bm_"
# A distinct kind per lifted method, so the dispatch pass treats their group as mixed: it
# splits into arms that call the real method, whatever the types the methods return.
KIND_ATTRIBUTE = "esbmc_kind"
_DYNAMIC = {"getattr", "setattr", "delattr", "hasattr", "vars", "locals", "globals", "eval", "exec"}
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _module_classes(module):
    return {s.name: s for s in module.body if isinstance(s, ast.ClassDef)}


def _is_local_to(fn, name):
    """Whether `name` is a parameter or an assigned name of `fn` that no `global` declaration lifts."""
    declared = any(
        isinstance(n, (ast.Global, ast.Nonlocal)) and name in n.names for n in _own_nodes(fn))
    params = [a.arg for a in ast.walk(fn.args) if isinstance(a, ast.arg)]
    stored = any(
        isinstance(n, ast.Name) and n.id == name and not isinstance(n.ctx, ast.Load)
        for n in _own_nodes(fn))
    return not declared and (name in params or stored)


def _module_scope_walk(node, name):
    """The nodes of `node` outside the functions in which `name` is a local of its own."""
    yield node
    if isinstance(node,
                  (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and _is_local_to(node, name):
        return
    for child in ast.iter_child_nodes(node):
        yield from _module_scope_walk(child, name)


def _binding_count(module, name):
    count = 0
    for node in _module_scope_walk(module, name):
        if isinstance(node, ast.Name) and node.id == name and not isinstance(node.ctx, ast.Load):
            count += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            count += node.name == name
        elif isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
            count += 2
        elif isinstance(node, ast.arg) and node.arg == name:
            count += 1
        elif isinstance(node, ast.alias) and (node.asname or node.name).split(".")[0] == name:
            count += 1
    return count


def _method_table(cls, classes):
    """The methods `cls` has by name along a single chain of module-level bases, else None."""
    table, seen = {}, set()
    while cls is not None:
        if id(cls) in seen or cls.keywords or cls.decorator_list or len(cls.bases) > 1:
            return None
        seen.add(id(cls))
        for member in cls.body:
            if isinstance(member, ast.FunctionDef):
                table.setdefault(member.name, member)
            elif isinstance(member, (ast.Assign, ast.AnnAssign, ast.AsyncFunctionDef)):
                return None
        if not cls.bases:
            return table
        base = cls.bases[0]
        cls = classes.get(base.id) if isinstance(base, ast.Name) else None
    return None


def _instances(module, classes):
    """`name -> (assignment, class)` for module-level `name = K(...)` of a module class, bound once."""
    found = {}
    for stmt in module.body:
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id in classes
                and not stmt.value.keywords):
            continue
        name = stmt.targets[0].id
        if _binding_count(module, name) == 1:
            found[name] = (stmt, classes[stmt.value.func.id])
    return found


def _is_scope_or_flow(node):
    return isinstance(node,
                      _SCOPES + (ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom, ast.Await))


def _is_self_method(value, table):
    return (isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name)
            and value.value.id == "self" and value.attr in table)


def _returned_methods(fn, table):
    """The names of the methods `fn` returns as `self.m` on every path, else None."""
    if not (_has_plain_signature(fn) and fn.args.args and fn.args.args[0].arg == "self"):
        return None
    names = []
    for node in _own_nodes(fn):
        if _is_scope_or_flow(node):
            return None
        if isinstance(node, ast.Return):
            if not _is_self_method(node.value, table):
                return None
            names.append(node.value.attr)
    return names if _always_returns(fn.body) and names else None


def _always_returns(block):
    if not block:
        return False
    last = block[-1]
    if isinstance(last, (ast.Return, ast.Raise)):
        return True
    return isinstance(last, ast.If) and _always_returns(last.body) and _always_returns(last.orelse)


def _self_is_contained(fn):
    """`self` is only read, and only as the receiver of an attribute."""
    parents = {}
    for node in ast.walk(fn):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id == "self" and not (
                isinstance(node.ctx, ast.Load) and isinstance(parents.get(id(node)), ast.Attribute)
                and parents[id(node)].value is node):
            if node is not fn.args.args[0] and not isinstance(node, ast.arg):
                return False
    return True


def _method_is_plain(method):
    args = method.args
    return not (method.decorator_list or args.vararg or args.kwarg or args.kwonlyargs
                or args.posonlyargs or args.defaults) and len(args.args) == 1


def _receiver_uses_are_attributes(module, name):
    """Every read of `name` is the receiver of an attribute: it is never passed on or stored."""
    parents = {}
    for node in ast.walk(module):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return all(
        isinstance(parents.get(id(n)), ast.Attribute) and parents[id(n)].value is n
        for n in _module_scope_walk(module, name)
        if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load))


def _stores_a_method_name(module, names):
    """Whether some attribute store or delete names one of the methods (`self.m = ...`)."""
    return any(
        isinstance(n, ast.Attribute) and n.attr in names and not isinstance(n.ctx, ast.Load)
        for n in ast.walk(module))


def _calls_of(module, receiver, method):
    """The `receiver.method(...)` call nodes, or None when `receiver.method` is read otherwise."""
    parents = {}
    for node in ast.walk(module):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    calls = []
    for node in _module_scope_walk(module, receiver):
        if isinstance(node, ast.Attribute) and node.attr == method and isinstance(
                node.value, ast.Name) and node.value.id == receiver:
            call = parents.get(id(node))
            if not (isinstance(call, ast.Call) and call.func is node):
                return None
            if not _is_bound_or_called(call, parents):
                return None
            calls.append(call)
    return calls


def _is_bound_or_called(call, parents):
    """The call is the whole value of an assignment to a name, or is itself called."""
    parent = parents.get(id(call))
    if isinstance(parent, ast.Call):
        return parent.func is call
    return isinstance(parent, ast.Assign) and parent.value is call and all(
        isinstance(t, ast.Name) for t in parent.targets)


def _wrapper(receiver, method, anchor):
    name = f"{BOUND}{receiver}_{method}"
    call = ast.Call(ast.Attribute(ast.Name(receiver, ast.Load()), method, ast.Load()), [], [])
    fn = ast.FunctionDef(name=name,
                         args=ast.arguments([], [], None, [], [], None, []),
                         body=[ast.Return(call)],
                         decorator_list=[],
                         returns=None,
                         type_comment=None,
                         type_params=[])
    setattr(fn, KIND_ATTRIBUTE, f"method:{receiver}.{method}")
    return ast.copy_location(fn, anchor)


class _ReadSelf(ast.NodeTransformer):
    """Read `self` as the receiver."""

    def __init__(self, receiver):
        self.receiver = receiver

    def visit_Name(self, node):
        if node.id == "self":
            return ast.copy_location(ast.Name(self.receiver, node.ctx), node)
        return node


class _BoundNames(_ReadSelf):
    """Also read the returned `self.m` as the lifted function."""

    def visit_Return(self, node):
        value = node.value
        return ast.copy_location(
            ast.Return(ast.Name(f"{BOUND}{self.receiver}_{value.attr}", ast.Load())), node)


def _lift_returner(module, receiver, method, fn, table, anchor):
    """The module-level function for `receiver.method`, and the names it returns."""
    import copy
    names = _returned_methods(fn, table)
    if names is None or len(set(names)) < 2 or not _self_is_contained(fn):
        return None
    if not all(_method_is_plain(table[m]) for m in set(names)):
        return None
    lifted = ast.FunctionDef(name=f"{BOUND}{receiver}_{method}",
                             args=ast.arguments([], [ast.arg(a.arg) for a in fn.args.args[1:]],
                                                None, [], [], None, []),
                             body=[_BoundNames(receiver).visit(copy.deepcopy(s)) for s in fn.body],
                             decorator_list=[],
                             returns=None,
                             type_comment=None,
                             type_params=[])
    return ast.copy_location(lifted, anchor), sorted(set(names))


def _bound_names(expr, receiver, table):
    """The method names of `expr` when it is `receiver.m` or a conditional of such, else None."""
    if isinstance(expr, ast.Attribute):
        ok = isinstance(expr.value, ast.Name) and expr.value.id == receiver and expr.attr in table
        return [expr.attr] if ok else None
    if isinstance(expr, ast.IfExp):
        body = _bound_names(expr.body, receiver, table)
        orelse = _bound_names(expr.orelse, receiver, table)
        return body + orelse if body and orelse else None
    return None


def _method_value_nodes(expr):
    if isinstance(expr, ast.IfExp):
        return _method_value_nodes(expr.body) + _method_value_nodes(expr.orelse)
    return [expr]


def _locals_of(fn):
    names = {a.arg for a in fn.args.args}
    names |= {
        n.id
        for n in ast.walk(fn) if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load)
    }
    return names


def _has_plain_signature(fn):
    args = fn.args
    return not (fn.decorator_list or args.vararg or args.kwarg or args.kwonlyargs
                or args.posonlyargs or args.defaults)


def _takes_self_and(fn, arity):
    return len(fn.args.args) == arity + 1 and fn.args.args[0].arg == "self"


def _has_scope_or_flow_node(fn):
    flow = _SCOPES + (ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom, ast.Await)
    return any(isinstance(n, flow) for n in _own_nodes(fn))


def _is_liftable(receiver, fn, arity):
    if not (_has_plain_signature(fn) and _takes_self_and(fn, arity)):
        return False
    if _has_scope_or_flow_node(fn) or not _self_is_contained(fn):
        return False
    uses_self = any(isinstance(n, ast.Name) and n.id == "self" for n in ast.walk(fn))
    return not (uses_self and receiver in _locals_of(fn))


def _lift_method(receiver, fn, arity, anchor):
    """`fn` as a module-level function with `self` read as `receiver`, or None."""
    import copy
    if not _is_liftable(receiver, fn, arity):
        return None
    args = fn.args
    lifted = ast.FunctionDef(name=f"{BOUND}{receiver}_{fn.name}",
                             args=ast.arguments([], [ast.arg(a.arg) for a in args.args[1:]], None,
                                                [], [], None, []),
                             body=[_ReadSelf(receiver).visit(copy.deepcopy(s)) for s in fn.body],
                             decorator_list=[],
                             returns=None,
                             type_comment=None,
                             type_params=[])
    setattr(lifted, KIND_ATTRIBUTE, f"method:{receiver}.{fn.name}")
    return ast.copy_location(lifted, anchor)


def _method_value_function(receiver, fn, anchor):
    if len(fn.args.args) == 1:
        return _wrapper(receiver, fn.name, anchor) if _method_is_plain(fn) else None
    return _lift_method(receiver, fn, len(fn.args.args) - 1, anchor)


def _argument_calls(module, receiver, table):
    """The `receiver.m(...)` calls with a bound method of `receiver` among their arguments."""
    calls = []
    for node in _module_scope_walk(module, receiver):
        func = getattr(node, "func", None)
        if not (isinstance(node, ast.Call) and isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name) and func.value.id == receiver
                and func.attr in table and not node.keywords):
            continue
        if any(_bound_names(a, receiver, table) for a in node.args):
            calls.append(node)
    return calls


def _lift_argument_calls(module, receiver, table, anchor):
    """Lift `receiver.m(..., receiver.f, ...)` to a call of module-level functions."""
    definitions, rewrites = {}, []
    for call in _argument_calls(module, receiver, table):
        callee = _lift_method(receiver, table[call.func.attr], len(call.args), anchor)
        names = {n for a in call.args for n in _bound_names(a, receiver, table) or []}
        values = {n: _method_value_function(receiver, table[n], anchor) for n in sorted(names)}
        if callee is None or None in values.values():
            continue
        definitions[callee.name] = callee
        definitions.update({v.name: v for v in values.values()})
        rewrites.append((call, callee.name))
    if not rewrites or any(_defined(module, name) for name in definitions):
        return []
    for call, name in rewrites:
        call.func = ast.copy_location(ast.Name(name, ast.Load()), call.func)
        call.args = [_rewrite_value(a, receiver, table) for a in call.args]
    return list(definitions.values())


def _rewrite_value(expr, receiver, table):
    if not _bound_names(expr, receiver, table):
        return expr
    if isinstance(expr, ast.IfExp):
        expr.body = _rewrite_value(expr.body, receiver, table)
        expr.orelse = _rewrite_value(expr.orelse, receiver, table)
        return expr
    return ast.copy_location(ast.Name(f"{BOUND}{receiver}_{expr.attr}", ast.Load()), expr)


def _defined(module, name):
    return any(isinstance(n, ast.FunctionDef) and n.name == name for n in ast.walk(module))


def _lift_returners(module, receiver, assign, table):
    returners = {m for m, fn in table.items() if _returned_methods(fn, table)}
    new = []
    for method in sorted(returners):
        calls = _calls_of(module, receiver, method)
        lifted = calls and _lift_returner(module, receiver, method, table[method], table, assign)
        if lifted:
            new.append((lifted[0], lifted[1], calls))
    needed = {m for _, flowing, _ in new for m in flowing}
    definitions = [_wrapper(receiver, m, assign) for m in sorted(needed)] + [fn for fn, _, _ in new]
    for fn, _, calls in new:
        for call in calls:
            call.func = ast.copy_location(ast.Name(fn.name, ast.Load()), call.func)
    return definitions


def lift_bound_methods(module):
    """Lift in place the bound methods that a module-level instance returns or is passed."""
    names = {n.id for n in ast.walk(module) if isinstance(n, ast.Name)}
    if names & _DYNAMIC or any(
            isinstance(n, ast.Attribute) and n.attr in ("__dict__", "__class__")
            for n in ast.walk(module)):
        return False
    classes = _module_classes(module)
    changed = False
    for receiver, (assign, cls) in _instances(module, classes).items():
        table = _method_table(cls, classes)
        if (table is None or not _receiver_uses_are_attributes(module, receiver)
                or _stores_a_method_name(module, set(table))):
            continue
        definitions = _lift_returners(module, receiver, assign, table)
        definitions += _lift_argument_calls(module, receiver, table, assign)
        position = module.body.index(assign) + 1
        module.body[position:position] = definitions
        changed = changed or bool(definitions)
    if changed:
        ast.fix_missing_locations(module)
    return changed
