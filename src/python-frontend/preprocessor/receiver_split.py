"""Split the module where a name is bound to instances of different classes (#10351).

A name holds one C type, so receiver_classes refuses a method call, an
attribute read, a store or a pass of a name that holds instances of different
classes on different paths. When the entry module binds such a name in the
branches of a module-level `if`, or by a conditional expression assigned to
it at module level, this pass copies the rest of the module into each branch,
as function_value_dispatch does for a call through a function value. Each
branch renames the names it binds, so each copy of the name holds instances of
one class and each value computed from it has that branch's type.

The module is left as it was, and the refusal stands, in these cases:

- the split does not clear the refusal
- a copy of the name is passed, stored or read other than as the receiver of
  an attribute, by isinstance, type or hasattr, in an identity test or to
  copy it to another name
- the branches call anything but a builtin, a class, a method with no
  arguments or a function that calls only builtins, since a lambda or a
  decorated function would see values of each branch's type
- a branch would rename a name that code before the branch reads or binds
- a branch defines a function, class or lambda, imports, declares a name
  global or reads names by reflection
- the module would need more than 32 copies
"""

import ast
import builtins
import copy
import itertools
import re

from preprocessor.binding_sites import binding_sites
from preprocessor.function_value_dispatch import _NESTED, _UNCOPYABLE, _visible_names
from preprocessor.function_value_types import _REFLECTIVE
from preprocessor.receiver_classes import (_module_classes, _own_nodes, _scope_receivers,
                                           refuse_path_dependent_receivers)
from preprocessor.unbound_locals import _located

_CAP = 32
_PREFIX = "_esbmc_rs"
_INSPECTORS = {"isinstance", "type", "hasattr"}


class _Refused(Exception):
    pass


def _refused(module, module_name):
    try:
        refuse_path_dependent_receivers(copy.deepcopy(module), module_name)
    except NotImplementedError:
        return True
    return False


def _as_if(target, value):
    """`target = value` with each conditional expression of `value` spelt as an `if`."""
    if not isinstance(value, ast.IfExp):
        return [ast.Assign(targets=[copy.deepcopy(target)], value=value)]
    return [
        ast.If(test=value.test,
               body=_as_if(target, value.body),
               orelse=_as_if(target, value.orelse))
    ]


def _base(name):
    """`name` without the prefix a branch gave it."""
    return re.sub(rf"^{_PREFIX}\d+_", "", name)


def _conditional_binding(stmt, receivers):
    """The `if` spelling of `stmt` when it assigns a conditional expression to a receiver."""
    if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name) and _base(stmt.targets[0].id) in receivers
            and isinstance(stmt.value, ast.IfExp)):
        return _located(_as_if(stmt.targets[0], stmt.value), stmt)[0]
    return stmt


def _binds(stmt, names):
    return any(
        isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load) and _base(n.id) in names
        for n in _own_nodes([stmt]))


def _stored(arm):
    """The names `arm` binds, or _Refused when it binds one other than by a plain name."""
    stored = set()
    for node in (n for s in arm for n in ast.walk(s)):
        if isinstance(node, _UNCOPYABLE):
            raise _Refused
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _REFLECTIVE | {"dir"}):
            raise _Refused
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            stored.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            stored.add(node.name)
    return stored


class _Splitter:

    def __init__(self, receivers):
        self.receivers = receivers
        self.arms = itertools.count(1)

    def block(self, stmts, before):
        """`stmts` split at its first `if` that binds a receiver.

        `before` holds the names that the code before `stmts` reads or binds.
        """
        for i, stmt in enumerate(stmts):
            stmt = _conditional_binding(stmt, self.receivers)
            if isinstance(stmt, ast.If) and _binds(stmt, self.receivers):
                before = before | _visible_names(stmts[:i] + [stmt.test])
                return stmts[:i] + [self._split(stmt, stmts[i + 1:], before)]
        return stmts

    def _split(self, stmt, rest, before):
        branches = []
        for branch in (stmt.body, stmt.orelse):
            arm = branch + copy.deepcopy(rest)
            self._rename(arm, before)
            branches.append(self.block(arm, before))
        return ast.copy_location(ast.If(test=stmt.test, body=branches[0], orelse=branches[1]), stmt)

    def _rename(self, arm, before):
        stored = _stored(arm)
        number = next(self.arms)
        if stored & before or number > _CAP:
            raise _Refused
        for node in (n for s in arm for n in ast.walk(s)):
            if isinstance(node, ast.Name) and node.id in stored:
                node.id = self._renamed(node.id, number)
            elif isinstance(node, ast.ExceptHandler) and node.name in stored:
                node.name = self._renamed(node.name, number)

    @staticmethod
    def _renamed(name, number):
        return f"{_PREFIX}{number}_{_base(name)}"


def _inspected(node, parent):
    """Whether `parent` reads `node` as the receiver of an attribute, by
    isinstance, type or hasattr, in an identity test, or to copy it to a name."""
    if isinstance(parent, ast.Attribute):
        return True
    if isinstance(parent, ast.Call):
        return (isinstance(parent.func, ast.Name) and parent.func.id in _INSPECTORS
                and node in parent.args)
    if isinstance(parent, ast.Compare):
        return all(isinstance(op, (ast.Is, ast.IsNot)) for op in parent.ops)
    return (isinstance(parent, ast.Assign) and len(parent.targets) == 1
            and isinstance(parent.targets[0], ast.Name))


def _copies(module, receivers):
    """The names of the split module that hold a copy of a receiver, directly
    or through a copy to another name."""
    copies = {
        n.id
        for n in ast.walk(module)
        if isinstance(n, ast.Name) and n.id.startswith(_PREFIX) and _base(n.id) in receivers
    }
    pairs = [(n.targets[0].id, n.value.id) for n in ast.walk(module)
             if isinstance(n, ast.Assign) and len(n.targets) == 1
             and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Name)]
    changed = True
    while changed:
        found = {target for target, value in pairs if value in copies}
        changed = not found <= copies
        copies |= found
    return copies


def _passes_receiver(module, receivers):
    """Whether the split module reads a copy of a receiver other than as
    `_inspected` allows. A parameter or an attribute would then hold
    instances of different classes, which the split does not separate."""
    copies = _copies(module, receivers)
    return any(
        isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in copies
        and not _inspected(node, parent) for parent in ast.walk(module)
        for node in ast.iter_child_nodes(parent))


def _is_builtin_call(call, bound):
    """Whether `call` calls a builtin or a `nondet_` intrinsic the module does
    not rebind, or a function of `_sv_verifier`."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id not in bound and (hasattr(builtins, func.id)
                                         or func.id.startswith("nondet_"))
    return isinstance(func, ast.Attribute) and isinstance(
        func.value, ast.Name) and func.value.id == "_sv_verifier"


def _plain_functions(module, bound):
    """The module-level functions that are undecorated, define nothing and
    call only builtins and `_sv_verifier`. The converter converts such a
    function again for arguments of other types."""
    return {
        fn.name
        for fn in module.body if isinstance(fn, ast.FunctionDef) and not fn.decorator_list
        and bound[fn.name] == 1 and not any(
            isinstance(n, _NESTED + (ast.Yield, ast.YieldFrom))
            for n in ast.walk(fn) if n is not fn) and all(
                _is_builtin_call(n, bound) for n in ast.walk(fn) if isinstance(n, ast.Call))
    }


def _bound_methods(module, copies):
    """The names bound to a method of a copy of a receiver."""
    return {
        n.targets[0].id
        for n in ast.walk(module) if isinstance(n, ast.Assign) and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Attribute)
        and isinstance(n.value.value, ast.Name) and n.value.value.id in copies
    }


def _calls_known(module, classes, receivers):
    """Whether every call in the split statement of `module` is of a builtin,
    `_sv_verifier`, a plain function, a class of `classes` or a method with no
    arguments. A lambda, a decorated function or a function value has one body
    for every call, so the split must not call one with values of each
    branch's type."""
    bound = binding_sites(module)
    plain = _plain_functions(module, bound) | set(classes)
    methods = _bound_methods(module, _copies(module, receivers))
    return all(
        _is_builtin_call(n, bound) or isinstance(n.func, ast.Name) and n.func.id in plain
        or not (n.args or n.keywords) and
        (isinstance(n.func, ast.Attribute) or isinstance(n.func, ast.Name) and n.func.id in methods)
        for n in _own_nodes(module.body[-1:]) if isinstance(n, ast.Call))


def split_path_dependent_receivers(module, module_name):
    """Split the entry module where it binds a receiver, when that clears the refusal."""
    classes, returns = _module_classes(module)
    if len(classes) < 2 or not _refused(module, module_name):
        return
    receivers = set(_scope_receivers(module, module, classes, returns))
    if not receivers:
        return
    candidate = copy.deepcopy(module)
    try:
        candidate.body = _Splitter(receivers).block(candidate.body, frozenset())
    except _Refused:
        return
    ast.fix_missing_locations(candidate)
    if (_calls_known(candidate, classes, receivers)
            and not (_passes_receiver(candidate, receivers) or _refused(candidate, module_name))):
        module.body = candidate.body
