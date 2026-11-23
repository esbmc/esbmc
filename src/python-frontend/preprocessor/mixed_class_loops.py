"""Unroll a `for` over a literal of instances of different classes (#10305).

A name holds one C type, the class it is first bound to, so a loop target
bound in turn to an instance of a class and of its subclass would call the
first class's methods on both. When a loop iterates a list or tuple literal,
or a name assigned one, whose elements are instances of at least two classes
of this module, this pass evaluates the elements in order into fresh names
and copies the body once per element, each copy reading its own name in
place of the target. Each copy is then typed by its own element, and the
target is bound to the last element after the copies.

The loop is left as it was, and receiver_classes refuses an attribute read
on the target that depends on the element, in these cases:

- the loop has an else, or its body breaks, continues, returns, yields,
  awaits or defines a function, class or lambda
- the target is not a plain name, is a parameter or declared global or
  nonlocal, or another scope reads it
- the scope reads names by reflection
- an element is starred, or the literal has more than 32 elements
- a name iterated is a parameter or global, is named in another scope, is not
  bound once, at the top level of the scope before the loop, to the literal,
  or is read other than as the iterable of such a loop
"""

import ast
import copy
import itertools

from preprocessor.function_value_types import _REFLECTIVE
from preprocessor.receiver_classes import (_assigned_literals, _binding_nodes, _bound_classes,
                                           _constructed_classes, _module_classes, _outside_names,
                                           _own_nodes, _parameters, _scope_blocks)
from preprocessor.unbound_locals import _located

_PREFIX = "_esbmc_mcl"
_CAP = 32
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
_EXITS = (ast.Break, ast.Continue, ast.Return, ast.Yield, ast.YieldFrom, ast.Await) + _SCOPES


def _read_elsewhere(name, scope):
    """Whether a function, lambda or class that `scope` encloses (any one,
    for the module) reads `name` from outside itself."""
    for node in ast.walk(scope):
        if node is scope or not isinstance(node, _SCOPES):
            continue
        if not any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node)):
            continue
        body = node.body if isinstance(node.body, list) else [node.body]
        params = set() if isinstance(node, ast.ClassDef) else _parameters(node)
        declared = _outside_names(body, set())
        if name in declared or (name not in params and not any(_binding_nodes(body, name))):
            return True
    return False


def _scope_ok(scope, module):
    """Whether `scope` reads no names by reflection, and the names it binds
    from outside it."""
    body = scope.body
    params = frozenset() if scope is module else _parameters(scope)
    reflective = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _REFLECTIVE
        for n in _own_nodes(body))
    return not reflective, _outside_names(body, params)


def _copyable(loop, scope, outside):
    """Whether the body of `loop` can be copied once per element, with its
    target renamed in each copy: no other scope reads the target."""
    if loop.orelse or not isinstance(loop.target, ast.Name):
        return False
    if any(isinstance(n, _EXITS) for stmt in loop.body for n in ast.walk(stmt)):
        return False
    name = loop.target.id
    return name not in outside and not _read_elsewhere(name, scope)


def _mixed(elts, bound, classes, returns):
    """Whether `elts` are instances of at least two classes of this module."""
    if len(elts) > _CAP or any(isinstance(e, ast.Starred) for e in elts):
        return False
    found = set()
    for elt in elts:
        if isinstance(elt, ast.Name):
            found |= bound.get(elt.id, set())
        else:
            found |= _constructed_classes(elt, classes, returns) or set()
    return len(found) > 1


def _assign(name, value):
    return ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=value)


def _unrolled(loop, fresh):
    """The statements replacing `loop`, whose iterable is a literal: the
    target is left bound to the last element, as the loop leaves it."""
    names = [f"{_PREFIX}{next(fresh)}" for _ in loop.iter.elts]
    stmts = _located([_assign(n, e) for n, e in zip(names, loop.iter.elts)], loop)
    for name in names:
        for stmt in copy.deepcopy(loop.body):
            for node in ast.walk(stmt):
                if isinstance(node, ast.Name) and node.id == loop.target.id:
                    node.id = name
            stmts.append(stmt)
    last = _assign(loop.target.id, ast.Name(id=names[-1], ctx=ast.Load()))
    return stmts + _located([last], loop)


def _read_other_than(body, name, reads):
    """Whether this scope reads `name` other than at the nodes `reads`."""
    return any(
        isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)
        and id(n) not in reads for n in _own_nodes(body))


def _sole_literal_binding(body, name):
    """The top-level statement of this scope that assigns a list or tuple
    literal without a starred element to `name`, when it is the scope's only
    binding of `name`; else None."""
    bindings = list(_binding_nodes(body, name))
    for stmt in body:
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [getattr(stmt, "target", None)]
        if (isinstance(stmt, (ast.Assign, ast.AnnAssign)) and len(bindings) == 1
                and targets == bindings and isinstance(stmt.value, (ast.List, ast.Tuple))
                and not any(isinstance(e, ast.Starred) for e in stmt.value.elts)):
            return stmt
    return None


def _inline_named_list(scope, name, loops, fresh):
    """Make each loop of `loops` iterate the literal assigned to `name`,
    through fresh names its binding now assigns the elements to, when that
    binding comes first and every other read of `name` is such a loop."""
    body = scope.body
    stmt = _sole_literal_binding(body, name)
    if (stmt is None or any(loop.lineno <= stmt.lineno for loop in loops)
            or _read_elsewhere(name, scope)
            or _read_other_than(body, name, {id(loop.iter)
                                             for loop in loops})):
        return False
    names = [f"{_PREFIX}{next(fresh)}" for _ in stmt.value.elts]
    position = body.index(stmt)
    body[position:position] = _located([_assign(n, e) for n, e in zip(names, stmt.value.elts)],
                                       stmt)
    stmt.value.elts = _located([ast.Name(id=n, ctx=ast.Load()) for n in names], stmt)
    for loop in loops:
        loop.iter = _located([ast.List(elts=copy.deepcopy(stmt.value.elts), ctx=ast.Load())],
                             loop.iter)[0]
    return True


def _innermost_candidates(scope, module, classes, returns):
    """The loops of `scope` this pass can unroll, without those that hold
    another such loop."""
    reflective_free, outside = _scope_ok(scope, module)
    if not reflective_free:
        return [], []
    bound = _bound_classes(scope.body, classes, returns)
    literals = _assigned_literals(scope.body)

    def elements(loop):
        if isinstance(loop.iter, (ast.List, ast.Tuple)):
            return loop.iter.elts
        named = isinstance(loop.iter, ast.Name) and loop.iter.id not in outside
        return literals.get(loop.iter.id, []) if named else []

    loops = [
        n for n in _own_nodes(scope.body) if isinstance(n, ast.For)
        and _mixed(elements(n), bound, classes, returns) and _copyable(n, scope, outside)
    ]
    return [
        loop for loop in loops
        if not any(other is not loop and other in ast.walk(loop) for other in loops)
    ], loops


def _unroll_one(scope, module, classes, returns, fresh):
    """Unroll one loop of `scope`; whether one was."""
    innermost, loops = _innermost_candidates(scope, module, classes, returns)
    for loop in innermost:
        if isinstance(loop.iter, ast.Name):
            same = [n for n in loops if isinstance(n.iter, ast.Name) and n.iter.id == loop.iter.id]
            if not _inline_named_list(scope, loop.iter.id, same, fresh):
                continue
        for block in _scope_blocks(scope.body):
            for position, stmt in enumerate(block):
                if stmt is loop:
                    block[position:position + 1] = _unrolled(loop, fresh)
                    return True
    return False


def unroll_mixed_class_loops(module):
    classes, returns = _module_classes(module)
    if len(classes) < 2:
        return
    fresh = itertools.count()
    scopes = [module] + [
        node
        for node in ast.walk(module) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for scope in scopes:
        while _unroll_one(scope, module, classes, returns, fresh):
            pass
