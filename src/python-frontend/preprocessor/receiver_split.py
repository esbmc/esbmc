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
from preprocessor.receiver_classes import (_module_classes, _own_nodes, _scope_blocks,
                                           _scope_receivers, refuse_path_dependent_receivers)
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


def _from_imported(module):
    """The dotted names a top-level `from m import x` may bind as modules."""
    return [
        f"{s.module}.{a.name}" for s in module.body
        if isinstance(s, ast.ImportFrom) and s.module and not s.level for a in s.names
    ]


def _distinct_imports(module):
    """Per name a top-level `import` binds and nothing else does, the module
    it binds; empty unless every import of `module` binds a module no other
    binds, so that two such names are one module only when they are one name.
    `from pkg import sub` may bind the module `pkg.sub`."""
    sites = binding_sites(module)
    imports = [a for s in module.body if isinstance(s, ast.Import) for a in s.names]
    names = {
        a.asname or a.name.split(".")[0]: a.name if a.asname else a.name.split(".")[0]
        for a in imports
    }
    bound = list(names.values()) + _from_imported(module)
    if len(set(bound)) != len(bound) or any(sites[name] != 1 for name in names):
        return {}
    return names


def _module_leaves(value, modules):
    """The names of `modules` that `value`, a name or a conditional of names,
    may evaluate to; None for anything else."""
    if isinstance(value, ast.Name):
        return {value.id} if value.id in modules else None
    if isinstance(value, ast.IfExp):
        body, orelse = _module_leaves(value.body, modules), _module_leaves(value.orelse, modules)
        return None if body is None or orelse is None else body | orelse
    return None


def _module_holders(module, modules):
    """The module-level names bound only to one of several `modules`."""
    sites = binding_sites(module)
    found = {}
    for node in _own_nodes(module.body):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            leaves = _module_leaves(node.value, modules)
            found.setdefault(node.targets[0].id, []).append(leaves)
    return {
        name
        for name, leaves in found.items()
        if None not in leaves and len(set().union(*leaves)) > 1 and sites[name] == len(leaves)
    }


def _held_modules(candidate, modules):
    """Per copy of a holder in the split `candidate`, the module its one
    binding assigns, when that binding opens its block and so runs before
    every read of the copy; each such binding becomes `pass`."""
    sites = binding_sites(candidate)
    held = {}
    for block in _scope_blocks(candidate.body):
        first = block[0] if block else None
        if (isinstance(first, ast.Assign) and len(first.targets) == 1
                and isinstance(first.targets[0], ast.Name)
                and first.targets[0].id.startswith(_PREFIX) and isinstance(first.value, ast.Name)
                and first.value.id in modules and sites[first.targets[0].id] == 1):
            held[first.targets[0].id] = first.value.id
            block[0] = ast.copy_location(ast.Pass(), first)
    return held


class _ModuleReads(ast.NodeTransformer):
    """Read each held copy as its module, and decide identity tests between
    modules, which are the same module only when they are the same name."""

    def __init__(self, held, modules):
        self.held = held
        self.modules = modules

    def visit_Name(self, node):  # pylint: disable=invalid-name,missing-function-docstring
        if isinstance(node.ctx, ast.Load) and node.id in self.held:
            return ast.copy_location(ast.Name(id=self.held[node.id], ctx=ast.Load()), node)
        return node

    def visit_Compare(self, node):  # pylint: disable=invalid-name,missing-function-docstring
        self.generic_visit(node)
        sides = [node.left] + node.comparators
        if not (len(node.ops) == 1 and isinstance(node.ops[0], (ast.Is, ast.IsNot))
                and all(isinstance(s, ast.Name) and s.id in self.modules for s in sides)):
            return node
        same = sides[0].id == sides[1].id
        return ast.copy_location(ast.Constant(same == isinstance(node.ops[0], ast.Is)), node)


def _module_reads_only(module, holders, modules):
    """Whether every read of a holder is the object of an attribute read or a
    side of an `is` or `is not` test with one of `modules`."""
    for parent in ast.walk(module):
        for node in ast.iter_child_nodes(parent):
            if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                    and node.id in holders):
                continue
            if isinstance(parent, ast.Attribute):
                continue
            sides = [parent.left] + parent.comparators if isinstance(parent, ast.Compare) else []
            if not (len(sides) == 2 and isinstance(parent.ops[0], (ast.Is, ast.IsNot)) and all(
                    isinstance(s, ast.Name) and (s.id in modules or s.id in holders)
                    for s in sides)):
                return False
    return True


def _split_module_holders(module, module_name, classes):
    """Split the entry module where it binds a name to one of several
    imported modules, and read each copy as its module (#10379). The
    converter holds no module in a variable."""
    modules = _distinct_imports(module)
    holders = _module_holders(module, modules)
    if not holders or not _module_reads_only(module, holders, modules):
        return
    candidate = copy.deepcopy(module)
    try:
        candidate.body = _Splitter(holders).block(candidate.body, frozenset())
    except _Refused:
        return
    held = _held_modules(candidate, modules)
    _ModuleReads(held, modules).visit(candidate)
    ast.fix_missing_locations(candidate)
    if (any(
            isinstance(n, ast.Name) and n.id.startswith(_PREFIX) and _base(n.id) in holders
            for n in ast.walk(candidate)) or not _calls_known(candidate, classes, set())
            or _refused(candidate, module_name)):
        return
    module.body = candidate.body


def _kinds(value, known, returns):
    """The builtin types `value` may evaluate to, or None when not known here.
    `known` holds the types of names bound only to values of known types, and
    `returns` the types of what functions return."""
    if isinstance(value, ast.Constant):
        return {type(value.value).__name__}
    if isinstance(value, ast.Name):
        return known.get(value.id)
    if isinstance(value, ast.IfExp):
        body, orelse = _kinds(value.body, known, returns), _kinds(value.orelse, known, returns)
        return None if body is None or orelse is None else body | orelse
    if not isinstance(value, ast.Call):
        return None
    name = getattr(value.func, "id", None) or getattr(value.func, "attr", "")
    if isinstance(value.func, ast.Name) and name in returns:
        return returns[name]
    if isinstance(value.func, ast.Name) and name in ("int", "float"):
        return {name}
    if name in ("nondet_int", "nondet_float"):
        return {name[len("nondet_"):]}
    return None


def _return_kinds(module):
    """The builtin types each plain function returns, when it ends in a
    `return` and every `return` in it gives a value of known type."""
    plain = _plain_functions(module, binding_sites(module))
    returns = {}
    for fn in module.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in plain and isinstance(
                fn.body[-1], ast.Return):
            kinds = [
                _kinds(r.value, {}, {}) if r.value else None for r in ast.walk(fn)
                if isinstance(r, ast.Return)
            ]
            if None not in kinds:
                returns[fn.name] = set().union(*kinds)
    return returns


def _number_kinds(module):
    """The builtin types of each module-level name bound only to values of
    known types, through at most one copy to another name."""
    bindings = {}
    for node in _own_nodes(module.body):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            bindings.setdefault(node.id, [])
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            bindings.setdefault(node.targets[0].id, []).append(node.value)
    returns = _return_kinds(module)
    known = {}
    for _ in range(2):
        known = {
            name: set().union(*kinds)
            for name, values in bindings.items()
            for kinds in [[_kinds(v, known, returns) for v in values]]
            if values and None not in kinds
        }
    return known


def _int_or_float_names(module):
    """The module-level names bound to an int on some paths and a float on
    others, all by one top-level statement. A binding outside it would stay
    on the module's own path, where the converter gives each binding its own
    symbol."""
    statements = {}
    for stmt in module.body:
        for node in _own_nodes([stmt]):
            if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                statements.setdefault(node.id, set()).add(id(stmt))
    return {
        name
        for name, kinds in _number_kinds(module).items()
        if kinds == {"int", "float"} and len(statements[name]) == 1
    }


def _reads_only_own_names(fn):
    """Whether `fn` reads no name but its own parameters and locals and the
    builtins, so that it cannot see a copy of a split name."""
    own = {a.arg for a in ast.walk(fn.args) if isinstance(a, ast.arg)}
    own |= {
        n.id
        for n in ast.walk(fn) if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load)
    }
    return all(n.id in own or hasattr(builtins, n.id) for n in ast.walk(fn)
               if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))


def _closed_functions(module, bound):
    """The plain functions of `module` that read only their own names."""
    plain = _plain_functions(module, bound)
    return {
        fn.name
        for fn in module.body
        if isinstance(fn, ast.FunctionDef) and fn.name in plain and _reads_only_own_names(fn)
    }


def _calls_nothing_split(call, bound, closed):
    """Whether `call` calls a builtin, `_sv_verifier` or a function of
    `closed` with no arguments, so that it sees no copy of a split name."""
    if _is_builtin_call(call, bound):
        return True
    return isinstance(call.func,
                      ast.Name) and call.func.id in closed and not (call.args or call.keywords)


def _numbers_only(module):
    """Whether the split statement of `module` calls only builtins,
    `_sv_verifier` and, with no arguments, plain functions that read only
    their own names, stores only into names and tests identity only between
    `type()` results. Anything else may see the int and the float copies as
    one value or one parameter."""
    bound = binding_sites(module)
    closed = _closed_functions(module, bound)
    for node in _own_nodes(module.body[-1:]):
        if isinstance(node, ast.Call) and not _calls_nothing_split(node, bound, closed):
            return False
        if isinstance(node, (ast.Attribute, ast.Subscript)) and not isinstance(node.ctx, ast.Load):
            return False
        if isinstance(node, ast.Compare) and any(
                isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops) and not all(
                    isinstance(e, ast.Call) and getattr(e.func, "id", None) == "type"
                    for e in [node.left] + node.comparators):
            return False
    return True


def _split(module, names):
    """`module` split at the first `if` that binds one of `names`, or None."""
    candidate = copy.deepcopy(module)
    try:
        candidate.body = _Splitter(names).block(candidate.body, frozenset())
    except _Refused:
        return None
    return ast.fix_missing_locations(candidate)


def split_path_dependent_receivers(module, module_name):
    """Split the entry module where it binds a receiver, when that clears the
    refusal, or else where it binds a name to an int or a float by path
    (#8973)."""
    classes, returns = _module_classes(module)
    receivers = set()
    if len(classes) >= 2 and _refused(module, module_name):
        receivers = set(_scope_receivers(module, module, classes, returns))
    if not receivers:
        _split_module_holders(module, module_name, classes)
    names = receivers or _int_or_float_names(module)
    candidate = names and _split(module, names)
    if not candidate or _refused(candidate, module_name):
        return
    if (_calls_known(candidate, classes, receivers) and not _passes_receiver(candidate, receivers)
            if receivers else _numbers_only(candidate)):
        module.body = candidate.body
