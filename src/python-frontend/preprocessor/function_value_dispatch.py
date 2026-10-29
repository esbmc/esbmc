"""Dispatch a call through a function value whose functions return different types (#8970).

The converter gives a call through a function value one return type, so
function_value_types refuses such a call when its functions disagree. For a
group of module-level functions, or functions that a top-level `from m import f`
binds, whose values only flow through module-level names, lists and tuples,
this pass rewrites the program so that no function value is left:

- each reference to a function of the group becomes its index in the group,
  so `h = f if c else g` becomes `h = 0 if c else 1`
- `h is f` becomes `h == 0` and `callable(h)` becomes `True`
- a statement calling `h()` becomes `if h == 0: ... elif ...: ... else: ...`,
  and each arm holds that statement and the rest of the module, with `h()`
  replaced by a direct call to that arm's function. A call in the body of an
  `if` first copies the rest of the module into both branches of that `if`.
- a module-level function that only returns the call `h()` of its parameter,
  as in `def call(h): return h()`, returns a direct call in each arm of such a
  chain instead, so nothing is copied.

Each arm then sees the precise type of its call's result. A name bound in an
arm is renamed per arm, so that it does not join values of different types.
When any condition does not hold, the module is left as it was and
function_value_types refuses the call as before.
"""

import ast
import copy
import itertools
import re

from preprocessor.function_value_lift import lift_nested_defs
from preprocessor.function_value_methods import lift_bound_methods
from preprocessor.function_value_types import _Analysis, _differ, _own_nodes

# The most copies of the rest of the module that one module may need.
_CAP = 32
_PREFIX = "_esbmc_fv"
_HOISTED = "_esbmc_hc"
# Marks left on nodes between checking a use and rewriting it.
_SITE = "esbmc_fv_site"
_CODE = "esbmc_fv_code"
_IDENTITY = "esbmc_fv_identity"
_CALLABLE = "esbmc_fv_callable"
_CHOSEN = "esbmc_fv_chosen"
_RETURN = "esbmc_fv_return"
# Builtins and attributes that reach module names other than by name.
_REFLECTIVE = {"globals", "locals", "vars", "eval", "exec", "__import__"}
_REFLECTIVE_ATTRS = {"modules", "__dict__", "__globals__"}
_NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.ListComp,
           ast.SetComp, ast.DictComp, ast.GeneratorExp)
# Statements that cannot be copied into an arm: their bindings are not plain names.
_UNCOPYABLE = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda, ast.Import,
               ast.ImportFrom, ast.Global, ast.Nonlocal)
# Statements whose body may run other than once, or not as a plain sequence.
_COMPOUND = tuple(
    getattr(ast, n)
    for n in ("For", "AsyncFor", "While", "With", "AsyncWith", "Try", "TryStar", "Match")
    if hasattr(ast, n))


class _Refused(Exception):
    """A condition for the rewrite does not hold."""


def _simple_callee(func):
    """A callee that is a module-level name, a call of one, or a conditional of names."""
    if isinstance(func, ast.Name):
        return True
    if isinstance(func, ast.Call):
        return _simple_callee(func.func)
    return isinstance(func, ast.IfExp) and all(
        isinstance(arm, ast.Name) for arm in (func.body, func.orelse))


def _whole_call(stmt):
    """The call that is all of `stmt`, an expression or an assignment to names; else None."""
    if isinstance(stmt, ast.Expr):
        value = stmt.value
    elif isinstance(stmt, ast.Assign) and all(isinstance(t, ast.Name) for t in stmt.targets):
        value = stmt.value
    else:
        return None
    return value if isinstance(value, ast.Call) and _simple_callee(value.func) else None


def _hoist_callee(call, names):
    """Bind the callee of `call`, when it is not a name, to a temporary; the statements, in order.

    Python evaluates a callee, arguments included, before the arguments of the call it
    is the callee of, so the temporaries keep the order when `call` is a whole statement.
    """
    before = _hoist_callee(call.func, names) if isinstance(call.func, ast.Call) else []
    if isinstance(call.func, ast.Name):
        return before
    temp = ast.Name(f"{_HOISTED}{next(names)}", ast.Store())
    ast.copy_location(temp, call)
    bind = ast.copy_location(ast.Assign([temp], call.func, None), call)
    call.func = ast.copy_location(ast.Name(temp.id, ast.Load()), call)
    return before + [bind]


def _has_no_params(fn):
    """Whether `fn` takes no argument of any kind."""
    args = fn.args
    return not (args.args or args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg)


def _returns_name(fn):
    """The name `fn` returns when its body is just `return name`, else None."""
    if len(fn.body) == 1 and isinstance(fn.body[0], ast.Return) and isinstance(
            fn.body[0].value, ast.Name):
        return fn.body[0].value.id
    return None


def _constant_returners(module):
    """The module-level defs `def r(): return t` that only return another such def `t`."""
    defs = {s.name: s for s in module.body if isinstance(s, ast.FunctionDef)}

    def sole(name):
        return name in defs and not defs[name].decorator_list and sum(
            _binds_name(s, name) for s in module.body) == 1

    found = {}
    for name, fn in defs.items():
        target = _returns_name(fn)
        if target is not None and _has_no_params(fn) and sole(name) and sole(target):
            found[name] = target
    return found


def _resolve_constant(call, constants):
    """Replace a callee `r()` by `t` where `def r(): return t`, which has no effect to keep."""
    inner = call.func
    if not isinstance(inner, ast.Call):
        return
    _resolve_constant(inner, constants)
    inner = call.func
    if isinstance(inner.func, ast.Name) and inner.func.id in constants and not inner.args \
            and not inner.keywords:
        call.func = ast.copy_location(ast.Name(constants[inner.func.id], ast.Load()), inner)


def _hoist_callees(module, wanted=None):
    """Give every callee that is a call or a conditional its own statement, at module level.

    With `wanted`, only a statement one of whose temporaries is in it is changed; the
    temporaries are numbered the same either way.
    """
    if any(name.startswith(_HOISTED) for name in _bound_names(module.body)):
        raise _Refused
    names = itertools.count()
    constants = _constant_returners(module)
    body = []
    for stmt in module.body:
        call = _whole_call(stmt)
        if call is not None:
            _resolve_constant(call, constants)
        if call is not None and not isinstance(call.func, ast.Name):
            trial = copy.deepcopy(stmt)
            hoisted = _hoist_callee(_whole_call(trial), names)
            if wanted is None or {h.targets[0].id for h in hoisted} & wanted:
                body.extend(hoisted)
                stmt = trial
        body.append(stmt)
    module.body = [
        stmt for stmt in body if not (isinstance(stmt, ast.FunctionDef) and stmt.name in constants
                                      and not _reads_outside(body, stmt))
    ]


def _reads_outside(body, fn):
    """Whether a statement of `body` other than `fn` mentions the name of `fn`."""
    return any(
        isinstance(n, ast.Name) and n.id == fn.name for stmt in body if stmt is not fn
        for n in ast.walk(stmt))


def _may_lift_methods(module):
    """Whether a method returns `self.name` or a call is passed an attribute, which the method lift may turn into a function."""
    return any(
        (isinstance(n, ast.Return) and isinstance(n.value, ast.Attribute)
         and isinstance(n.value.value, ast.Name) and n.value.value.id == "self") or
        (isinstance(n, ast.Call) and any(isinstance(a, (ast.Attribute, ast.IfExp)) for a in n.args))
        for n in ast.walk(module))


def _mixed_groups(analysis):
    """The groups that a call goes through and whose functions disagree on return type."""
    roots = []
    for entry in analysis.calls:
        root = analysis.find(entry[1])
        if entry[-1] is None or root in roots:
            continue
        kinds = [kind for _, _, kind in analysis.members.get(root, {}).values() if kind]
        if any(_differ(a, b) for a in kinds for b in kinds):
            roots.append(root)
    return roots


def _dispatch_groups(analysis):
    """The mixed groups, and the groups of the defs that return into them.

    `func()()` goes through two groups: the defs `func` may return, which return the functions
    of a mixed group, and those functions. The first group needs codes too, for a call through
    one of its values to be a value of the second.
    """
    roots = _mixed_groups(analysis)
    for root in list(roots):
        returners = {
            key[1]
            for key in analysis.parent if key[0] == "result" and analysis.find(key) == root
        }
        for other, members in analysis.members.items():
            if other not in roots and any(id(m[0]) in returners for m in members.values()):
                roots.append(other)
    return roots


def _nested_members(module, analysis, roots):
    """The `(outer, inner)` names of the functions in `roots`' groups that a def nests."""
    return {(outer.name, member[0].name)
            for root in roots
            for member in analysis.members[root].values()
            for outer in module.body
            if isinstance(outer, ast.FunctionDef) and any(member[0] is s for s in outer.body)}


def _parents(module):
    parents = {}
    for node in ast.walk(module):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def _rebinds(module, name):
    """Whether `module` binds `name` anywhere."""
    for node in ast.walk(module):
        if isinstance(node, ast.Name) and node.id == name and not isinstance(node.ctx, ast.Load):
            return True
        if name in (getattr(node, "arg", None), getattr(node, "name",
                                                        None), getattr(node, "asname", None)):
            return True
    return False


def _binds_by_string(node, names):
    """Whether `node` itself binds one of `names` other than through an `ast.Name`."""
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return bool(set(node.names) & names)
    if isinstance(node, ast.alias):
        return (node.asname or node.name.split(".")[0]) in names
    bound = getattr(node, "name", None)
    if isinstance(node, ast.arg):
        bound = node.arg
    return isinstance(bound, str) and bound in names


def _bound_names(nodes):
    """Every name that `nodes` read or bind, in any scope."""
    names = set()
    for top in nodes:
        for node in ast.walk(top):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
            elif isinstance(node, ast.alias):
                names.add(node.asname or node.name.split(".")[0])
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                names.update(node.names)
            elif isinstance(getattr(node, "name", None), str):
                names.add(node.name)
    return names


def _scope_locals(scope):
    """The names a def, lambda or class binds for itself: parameters and its own bindings."""
    args = getattr(scope, "args", None)
    names = set()
    if args is not None:
        names.update(a.arg for a in args.posonlyargs + args.args + args.kwonlyargs)
        names.update(a.arg for a in (args.vararg, args.kwarg) if a is not None)
    declared = set()
    for node in _own_nodes(scope):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            declared.update(node.names)
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[0])
    return names - declared


_DEFINING = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


class _VisibleNames:
    """The names that nodes read or bind where the module can see them.

    A parameter or a local of a nested def is its own variable, so it does not count; a name
    that such a def declares `global` or `nonlocal`, or reads without binding, does.
    """

    def __init__(self):
        self.names = set()

    def visit(self, node, local):
        if isinstance(node, _DEFINING):
            self._definition(node, local)
            return
        self._own(node, local)
        for child in ast.iter_child_nodes(node):
            self.visit(child, local)

    def _definition(self, node, local):
        """A def, lambda or class: its decorators and defaults belong to the enclosing scope."""
        lam = isinstance(node, ast.Lambda)
        if not lam and node.name not in local:
            self.names.add(node.name)
        outside = [] if lam else list(node.decorator_list)
        if not isinstance(node, ast.ClassDef):
            outside += node.args.defaults + [d for d in node.args.kw_defaults if d is not None]
        for child in outside:
            self.visit(child, local)
        for child in [node.body] if lam else node.body:
            self.visit(child, local | _scope_locals(node))

    def _own(self, node, local):
        if isinstance(node, ast.Name) and node.id not in local:
            self.names.add(node.id)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            self.names.update(node.names)
        elif isinstance(node, ast.alias):
            self.names.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name and node.name not in local:
            self.names.add(node.name)


def _visible_names(nodes):
    visible = _VisibleNames()
    for node in nodes:
        visible.visit(node, frozenset())
    return visible.names


def _sole_def(fn, defs, module, analysis):
    """Whether `fn` is the module's only, undecorated def of its name, or the one that a
    single top-level `from m import name` binds there, and is never rebound."""
    name = getattr(fn, "name", None)
    if name is None or fn.decorator_list or any(
            isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store)
            for n in _own_nodes(module)):
        return False
    if defs.get(name) is fn:
        return len(analysis.defs.get((id(module), name), [])) == 1
    return _sole_import(fn, module, analysis)


def _sole_import(fn, module, analysis):
    """Whether `fn` is what the only binding of its name in `module` imports."""
    if not (len(_from_import_aliases(ast.walk(module), fn.name)) == 1
            and analysis.from_imported_def(fn.name, analysis.scopes[id(module)]) is fn):
        return False
    source = next(tree for tree in analysis.loaded.values() if fn in tree.body)
    return sum(_binds_name(stmt, fn.name) for stmt in source.body) == 1


def _from_import_aliases(nodes, name):
    """The aliases of the `from m import` statements among `nodes` that bind `name`."""
    return [
        alias for node in nodes if isinstance(node, ast.ImportFrom) for alias in node.names
        if (alias.asname or alias.name) == name
    ]


def _binds_name(stmt, name):
    """Whether the top-level `stmt` binds `name`, anywhere inside it."""
    return any(
        _binds_by_string(node, {name}) or (
            isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Store))
        for node in ast.walk(stmt))


def _takes_positional(fn, count):
    """Whether `fn` takes exactly `count` positional arguments and nothing else."""
    args = fn.args
    return not (args.vararg or args.kwarg or args.kwonlyargs or args.defaults
                or args.posonlyargs) and len(args.args) == count


def _always_returns(block):
    """Whether every path through `block` ends in a `return` or a `raise`."""
    if not block:
        return False
    last = block[-1]
    if isinstance(last, (ast.Return, ast.Raise)):
        return True
    return isinstance(last, ast.If) and _always_returns(last.body) and _always_returns(last.orelse)


def _is_callee(parent, node):
    return isinstance(parent, ast.Call) and parent.func is node


def _is_builtin_call(parent, name, arg):
    """Whether `parent` is `name(arg)` with no other argument."""
    return isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and \
        parent.func.id == name and parent.args == [arg] and not parent.keywords


def _call_directly(stmt, group, callee, fn):
    """Call `fn` at the chosen site in `stmt`, and at every site through the same name."""
    same = ast.dump(callee) if isinstance(callee, ast.Name) else None
    for node in ast.walk(stmt):
        if getattr(node, _SITE, None) == group and (getattr(node, _CHOSEN, False)
                                                    or ast.dump(node.func) == same):
            node.func = ast.copy_location(ast.Name(fn.name, ast.Load()), node.func)
            setattr(node, _SITE, None)


def _chain(arms, callee, origin):
    """`if callee == 0: arms[0] elif ...: ... else: arms[-1]`, located at `origin`."""
    chain = arms[-1]
    for code in range(len(arms) - 2, -1, -1):
        test = ast.Compare(copy.deepcopy(callee), [ast.Eq()], [ast.Constant(code)])
        chain = [ast.If(test, arms[code], chain)]
    for node in chain:
        ast.copy_location(node, origin)
    return chain


class _Plan:
    """Check that every use of the groups' names can be rewritten, and mark them."""

    def __init__(self, module, analysis, roots, lifted=frozenset()):
        self.module = module
        self.parents = _parents(module)
        self.functions = {}
        self.groups = []
        self.holders = {}
        self.param_holders = {}
        self.returners = {}
        self.returned_names = set()
        self.local_fns = set()
        self.imported = set()
        defs = {stmt.name: stmt for stmt in module.body if isinstance(stmt, ast.FunctionDef)}
        self.defs = defs
        for root in roots:
            group = len(self.groups)
            fns = sorted((m[0] for m in analysis.members[root].values()), key=lambda f: f.lineno)
            if not all(_sole_def(fn, defs, module, analysis) for fn in fns):
                raise _Refused
            self.groups.append(fns)
            self.imported.update(
                id(alias) for fn in fns if defs.get(fn.name) is not fn
                for alias in _from_import_aliases(module.body, fn.name))
            for code, fn in enumerate(fns):
                self.functions[fn.name] = (group, code)
            self._add_holders(analysis, root, group, defs)
        self.lifted_groups = {self.functions[name][0] for name in lifted if name in self.functions}
        self.callable_names = set(self.functions) | set(self.param_holders)
        self.kinds = {}
        self.values = {}
        self._collect_bindings()
        for name in self.holders:
            self._kind(name, set())

    def _add_holders(self, analysis, root, group, defs):
        """Record the module names and parameters in `root`'s group; refuse any other value."""
        by_id = {id(fn): fn for fn in defs.values()}
        for key in [k for k in analysis.parent if analysis.find(k) == root]:
            if key[0] in ("elements", "call"):
                continue
            if key[0] == "result":
                self._add_returner(by_id.get(key[1]), group)
                continue
            if key[0] == "name" and key[1] in by_id:
                self._add_param_holder(analysis, by_id[key[1]], key[2], group)
                continue
            if key[0] != "name" or key[1] != id(self.module):
                raise _Refused
            if key[2] not in self.functions:
                self.holders[key[2]] = group

    def _add_returner(self, fn, group):
        """Record a module-level def whose every return is a function of `group`."""
        if fn is None or fn.name in self.returners:
            raise _Refused
        self.returners[fn.name] = group

    def _add_param_holder(self, analysis, fn, param, group):
        """Record a parameter or local of a module-level def, which the def may dispatch on."""
        scope = analysis.scopes[id(fn)]
        if param not in scope.names or not _sole_def(fn, self.defs, self.module, analysis):
            raise _Refused
        if param not in scope.params:
            self.local_fns.add(id(fn))
        self.param_holders.setdefault(fn.name, {})[param] = group

    # bindings of holders
    def _collect_bindings(self):
        self.values = {name: [] for name in self.holders}
        for stmt in ast.walk(self.module):
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                self._bind(target, stmt.value)

    def _bind(self, target, value):
        if isinstance(target, ast.Name):
            if target.id in self.holders:
                self.values[target.id].append(value)
        elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(
                value, (ast.Tuple, ast.List)) and len(target.elts) == len(value.elts):
            for sub, elt in zip(target.elts, value.elts):
                self._bind(sub, elt)

    def _kind(self, name, seen):
        """How deep `name` nests functions: 0 for a function, 1 for a list or dict of them."""
        if name in self.kinds:
            return self.kinds[name]
        if name in seen or not self.values[name]:
            raise _Refused
        seen.add(name)
        kinds = {self._value_kind(value, seen) for value in self.values[name]}
        if len(kinds) != 1:
            raise _Refused
        self.kinds[name] = kinds.pop()
        return self.kinds[name]

    def _value_kind(self, expr, seen):
        if isinstance(expr, ast.Name):
            if expr.id in self.functions:
                return 0
            if expr.id in self.holders:
                return self._kind(expr.id, seen)
        elif isinstance(expr, ast.IfExp):
            kinds = {self._value_kind(expr.body, seen), self._value_kind(expr.orelse, seen)}
            if len(kinds) == 1:
                return kinds.pop()
        elif isinstance(expr, (ast.List, ast.Tuple, ast.Dict)):
            return self._literal_kind(expr, seen)
        elif self._is_element(expr):
            depth = self._value_kind(expr.value, seen)
            if depth > 0:
                return depth - 1
        elif self._is_returner_call(expr):
            return 0
        raise _Refused

    def _literal_kind(self, literal, seen):
        items = literal.values if isinstance(literal, ast.Dict) else literal.elts
        if not items or None in getattr(literal, "keys", []):
            raise _Refused
        kinds = {self._value_kind(item, seen) for item in items}
        if len(kinds) != 1:
            raise _Refused
        return kinds.pop() + 1

    def _call_result_group(self, expr):
        """The group of the functions that call `expr` returns, or None.

        `expr` calls a returner, or a holder all of whose functions are returners into one group.
        """
        if not (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)):
            return None
        name = expr.func.id
        if name in self.returners:
            return self.returners[name]
        if name not in self.holders or self.kinds.get(name, 0) != 0:
            return None
        groups = {self.returners.get(fn.name) for fn in self.groups[self.holders[name]]}
        return groups.pop() if len(groups) == 1 else None

    def _is_returner_call(self, expr):
        return self._call_result_group(expr) is not None

    @classmethod
    def _is_element(cls, expr):
        """`xs[i]`, possibly nested, with a name or literal index, whose evaluation has no effects."""
        if not isinstance(expr, ast.Subscript):
            return False
        if not (isinstance(expr.value, ast.Name) or cls._is_element(expr.value)):
            return False
        index = expr.slice
        if isinstance(index, ast.UnaryOp) and isinstance(index.op, ast.USub):
            index = index.operand
        return isinstance(index, (ast.Name, ast.Constant))

    def _element_kind(self, expr):
        """How deep the value of the element `expr` nests functions, or None."""
        depth = 0
        while isinstance(expr, ast.Subscript):
            expr, depth = expr.value, depth + 1
        kind = self.kinds.get(expr.id)
        return None if kind is None else kind - depth

    # uses
    def check_uses(self):
        names = set(self.functions) | set(self.holders) | set(self.param_holders)
        names |= set(self.returners)
        for name in self.returners:
            self._check_returner(self.defs[name], self.returners[name])
        bound = _bound_names([self.module])
        if bound & _REFLECTIVE or any(n.startswith(_PREFIX) for n in bound):
            raise _Refused
        for node in ast.walk(self.module):
            if isinstance(node, ast.Attribute) and node.attr in _REFLECTIVE_ATTRS:
                raise _Refused
            if isinstance(node, ast.Name):
                if node.id in names:
                    self._check_use(node)
            elif self._binds_group_name(node, names):
                raise _Refused
            elif isinstance(node, ast.FunctionDef) and node.name in self.holders:
                raise _Refused

    def _binds_group_name(self, node, names):
        """Whether `node` binds one of `names` other than through an `ast.Name`."""
        return node is not self.module and not isinstance(node, ast.FunctionDef) and \
            bool(_bound_names([node]) & names) and _binds_by_string(node, names) and \
            id(node) not in self.imported

    def _nested(self, node):
        parent = self.parents.get(id(node))
        while parent is not None:
            if isinstance(parent, _NESTED) and id(parent) not in self.local_fns:
                return True
            parent = self.parents.get(id(parent))
        return False

    def _is_dispatched_param(self, fn, expr, group):
        """Whether `expr` names a parameter of `fn` that `fn` dispatches on, in `group`."""
        return isinstance(expr, ast.Name) and self.param_holders.get(fn.name, {}).get(
            expr.id) == group and expr.id in {a.arg
                                              for a in fn.args.posonlyargs + fn.args.args}

    def _check_returner(self, fn, group):
        """`fn` has no nested scope and returns a function of `group` on every path."""
        body = ast.Module(fn.body, [])
        if fn.decorator_list or not _always_returns(fn.body) or any(
                isinstance(
                    n, _NESTED + (ast.Yield, ast.YieldFrom, ast.Await, ast.Global, ast.Nonlocal))
                for n in ast.walk(body)):
            raise _Refused
        for node in ast.walk(body):
            if isinstance(node, ast.Return):
                self._check_returned_function(node.value, group, fn)

    def _check_returned_function(self, expr, group, fn):
        if isinstance(expr, ast.IfExp):
            self._check_returned_function(expr.body, group, fn)
            self._check_returned_function(expr.orelse, group, fn)
        elif self._is_dispatched_param(fn, expr, group):
            self.returned_names.add(id(expr))
        elif isinstance(expr, ast.Name) and self.functions.get(expr.id, (None, ))[0] == group:
            setattr(expr, _CODE, self.functions[expr.id][1])
            self.returned_names.add(id(expr))
        else:
            raise _Refused

    def _check_returner_use(self, node, parent):
        """A returner is only called, at module level, for a value that a code can stand for."""
        if not (_is_callee(parent, node) and not self._nested(node)):
            raise _Refused
        self._check_value(parent)

    def _check_use(self, node):
        parent = self.parents[id(node)]
        if id(node) in self.returned_names:
            return
        if node.id in self.returners and (node.id not in self.functions
                                          or _is_callee(parent, node)):
            self._check_returner_use(node, parent)
            return
        if _is_callee(parent, node) and node.id in self.callable_names:
            return
        if self._nested(node) or node.id in self.param_holders:
            if not self._is_returned_read(node):
                raise _Refused
            return
        if isinstance(node.ctx, ast.Store):
            self._check_store(node)
        elif isinstance(node.ctx, ast.Del):
            raise _Refused
        elif node.id in self.functions:
            self._check_value(node)
            setattr(node, _CODE, self.functions[node.id][1])
        elif self.kinds[node.id] == 0:
            self._check_function_holder(node, parent)
        else:
            self._check_container_holder(node, parent)

    def _is_returned_read(self, node):
        """A module holder read in a flat module-level def only as the callee of its `return`."""
        fn = self._enclosing_def(node)
        if fn is None or node.id not in self.holders or not isinstance(node.ctx, ast.Load):
            return False
        if not self._is_flat(fn, node.id):
            return False
        top = node
        while isinstance(self.parents[id(top)], ast.Subscript) and \
                self.parents[id(top)].value is top:
            top = self.parents[id(top)]
        call = self.parents[id(top)]
        if not self._is_returned_callee(top, call, node):
            return False
        self._check_arguments(call, self.holders[self._site_holder(call)])
        setattr(self.parents[id(call)], _RETURN, self.holders[node.id])
        return True

    def _is_flat(self, fn, name):
        """Whether `fn` has no nested scope and does not bind `name`."""
        body = ast.Module(fn.body, [])
        args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        return not (any(isinstance(n, _NESTED) for n in ast.walk(body)) or _rebinds(body, name)
                    or name in {a.arg
                                for a in args})

    def _is_returned_callee(self, top, call, node):
        ret = self.parents[id(call)]
        return _is_callee(call, top) and isinstance(ret, ast.Return) and ret.value is call and \
            (top is node or self._is_element(top)) and self._element_kind(top) == 0 and \
            isinstance(top.ctx, ast.Load)

    def _enclosing_def(self, node):
        """The module-level def that contains `node` with no scope in between, or None."""
        parent = self.parents.get(id(node))
        while parent is not None and not isinstance(parent, _NESTED):
            parent = self.parents.get(id(parent))
        return parent if isinstance(parent, ast.FunctionDef) and self.defs.get(
            parent.name) is parent else None

    def check_imported(self):
        """An imported function is dispatched at module level only, where its result keeps its type."""
        if self.imported and (self.param_holders
                              or any(hasattr(n, _RETURN) for n in ast.walk(self.module))):
            raise _Refused

    def check_param_holders(self):
        """Each def with a dispatched parameter or local uses it only where it can be a code."""
        names = set(self.functions) | set(self.holders)
        for name, variables in self.param_holders.items():
            fn = self.defs[name]
            nodes = list(ast.walk(ast.Module(fn.body, [])))
            if set(variables) & names or any(isinstance(n, _NESTED) for n in nodes):
                raise _Refused
            params = {a.arg for a in fn.args.posonlyargs + fn.args.args}
            for var, group in variables.items():
                self._check_variable(var, var in params, group, nodes)
        for call in ast.walk(self.module):
            if isinstance(call, ast.Call) and getattr(call.func, "id", None) in self.param_holders:
                self._check_dispatched_arguments(call)

    def _check_variable(self, var, is_param, group, nodes):
        for node in nodes:
            if isinstance(node, (ast.Global, ast.Nonlocal)) and var in node.names:
                raise _Refused
            if var in (getattr(node, "name", None), getattr(node, "asname", None)):
                raise _Refused
            if isinstance(node, ast.Name) and node.id == var:
                self._check_variable_use(node, group, is_param)

    def _check_variable_use(self, use, group, is_param):
        """A load is a returned call or an argument to dispatch on; a store assigns a function."""
        if id(use) in self.returned_names:
            return
        parent = self.parents[id(use)]
        if isinstance(use.ctx, ast.Load):
            if not self._is_dispatched_argument(parent, use):
                self._check_returned_call(use, group)
        elif is_param or not (isinstance(parent, ast.Assign) and use in parent.targets
                              and self._is_function_value(parent.value, parent)):
            raise _Refused

    def _check_dispatched_arguments(self, call):
        """Every parameter of the callee that is dispatched on gets a function positionally."""
        fn = self.defs[call.func.id]
        holders = self.param_holders[fn.name]
        if any(isinstance(a, ast.Starred)
               for a in call.args) or any(k.arg is None or k.arg in holders for k in call.keywords):
            raise _Refused
        for i, param in enumerate(a.arg for a in fn.args.posonlyargs + fn.args.args):
            if param in holders and not (i < len(call.args)
                                         and self._is_function_value(call.args[i], call)):
                raise _Refused

    def _is_function_value(self, expr, site):
        """Whether `expr`, evaluated at `site`, is a function or a dispatched variable of its def."""
        if isinstance(expr, ast.IfExp):
            return self._is_function_value(expr.body, site) and self._is_function_value(
                expr.orelse, site)
        return self._is_code(expr) or (isinstance(expr, ast.Name)
                                       and expr.id in self._variables_at(site))

    def _variables_at(self, node):
        """The dispatched variables of the module-level def that contains `node`."""
        parent = self.parents.get(id(node))
        while parent is not None:
            if isinstance(parent, ast.FunctionDef):
                return self.param_holders.get(parent.name, {}) if self.defs.get(
                    parent.name) is parent else {}
            parent = self.parents.get(id(parent))
        return {}

    def _check_returned_call(self, use, group):
        """`use` is the callee of a call that is a whole `return`; mark that return."""
        call = self.parents[id(use)]
        ret = self.parents[id(call)]
        if not (_is_callee(call, use) and isinstance(ret, ast.Return) and ret.value is call):
            raise _Refused
        self._check_arguments(call, group)
        setattr(ret, _RETURN, group)

    def _check_store(self, node):
        if node.id in self.functions or not self._assign_target(node):
            raise _Refused

    def _check_function_holder(self, node, parent):
        if _is_callee(parent, node):
            self._check_site(parent)
            if self._is_returner_call(parent):
                self._check_value(parent)
        else:
            self._check_value(node)

    def _check_container_holder(self, node, parent):
        if isinstance(parent, ast.Subscript) and parent.value is node:
            self._check_element(parent)
        elif not (_is_builtin_call(parent, "len", node) or self._is_plain_value(parent, node)
                  or self._is_membership(parent, node) or self._is_update(parent, node)):
            raise _Refused

    def _check_element(self, element):
        """The outermost element `xs[i]...` is a store target, a callee or a value."""
        while isinstance(self.parents[id(element)], ast.Subscript) and \
                self.parents[id(element)].value is element:
            element = self.parents[id(element)]
        if not self._is_element(element):
            raise _Refused
        outer = self.parents[id(element)]
        if isinstance(element.ctx, ast.Store):
            self._check_element_store(element, outer)
        elif isinstance(element.ctx, ast.Del):
            if not isinstance(outer, ast.Delete):
                raise _Refused
        elif not isinstance(element.ctx, ast.Load):
            raise _Refused
        elif _is_callee(outer, element) and self._element_kind(element) == 0:
            self._check_site(outer)
        else:
            self._check_value(element)

    def _check_element_store(self, element, assign):
        """`xs[i] = value` stores a value as deep as the elements of `xs`."""
        if not (isinstance(assign, ast.Assign) and assign.targets == [element]):
            raise _Refused
        if self._value_kind(assign.value, set()) != self._element_kind(element):
            raise _Refused

    def _is_membership(self, parent, node):
        return isinstance(parent, ast.Compare) and len(parent.ops) == 1 and isinstance(
            parent.ops[0], (ast.In, ast.NotIn)) and parent.comparators[0] is node

    def _is_update(self, attribute, node):
        """`xs.update(literal)` or `xs.setdefault(key, value)` as a statement, deep as `xs`."""
        if not (isinstance(attribute, ast.Attribute) and attribute.value is node):
            return False
        call = self.parents[id(attribute)]
        if not (isinstance(call, ast.Call) and call.func is attribute and not call.keywords
                and isinstance(self.parents[id(call)], ast.Expr)):
            return False
        if attribute.attr == "update":
            return len(call.args) == 1 and isinstance(call.args[0], ast.Dict) and \
                self._value_kind(call.args[0], set()) == self.kinds[node.id]
        return attribute.attr == "setdefault" and len(call.args) == 2 and \
            self._value_kind(call.args[1], set()) == self.kinds[node.id] - 1

    def _assign_target(self, node):
        """Whether `node` is bound by an assignment whose value for it can be read off."""
        path = []
        parent = self.parents[id(node)]
        child = node
        while isinstance(parent, (ast.Tuple, ast.List)):
            path.append(parent.elts.index(child))
            child, parent = parent, self.parents[id(parent)]
        if not isinstance(parent, ast.Assign):
            return False
        value = parent.value
        target = child
        for index in reversed(path):
            if not (isinstance(value,
                               (ast.Tuple, ast.List)) and len(value.elts) == len(target.elts)):
                return False
            value, target = value.elts[index], target.elts[index]
        return not any(isinstance(n, ast.Starred) for t in parent.targets for n in ast.walk(t))

    def _plain_targets(self, assign):
        """Whether every target of `assign` is a name, an element or a tuple of them."""
        return all(self._plain_target(t) for t in assign.targets)

    def _plain_target(self, target):
        if isinstance(target, (ast.Tuple, ast.List)):
            return all(self._plain_target(t) for t in target.elts)
        return isinstance(target, ast.Name) or self._is_element(target)

    def _check_value(self, expr):
        """`expr`, a function or an element, flows only where a code can stand for it."""
        parent = self.parents[id(expr)]
        if isinstance(parent, ast.IfExp) and expr is not parent.test:
            self._check_value(parent)
        elif isinstance(parent, (ast.List, ast.Tuple)) or self._is_dict_value(parent, expr):
            self._check_value_list(parent)
        elif self._is_plain_value(parent, expr) or self._is_dispatched_argument(parent, expr) \
                or self._is_update_value(parent, expr):
            pass
        elif isinstance(parent, ast.Compare):
            self._check_identity(parent)
        elif _is_builtin_call(parent, "callable", expr) and not _rebinds(self.module, "callable"):
            setattr(parent, _CALLABLE, True)
        else:
            raise _Refused

    def _is_dispatched_argument(self, parent, expr):
        """Whether `expr` is passed positionally to a parameter that the callee dispatches on."""
        if not (isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name)
                and parent.func.id in self.param_holders and expr in parent.args
                and not any(isinstance(a, ast.Starred) for a in parent.args)):
            return False
        fn = self.defs[parent.func.id]
        params = fn.args.posonlyargs + fn.args.args
        index = parent.args.index(expr)
        return index < len(params) and params[index].arg in self.param_holders[fn.name]

    def _is_plain_value(self, parent, expr):
        """Whether `expr` is the whole value of an assignment to names."""
        return isinstance(parent, ast.Assign) and parent.value is expr and \
            self._plain_targets(parent)

    def _check_identity(self, compare):
        """`compare` tests two codes for identity or equality."""
        if len(compare.ops) != 1 or not isinstance(compare.ops[0],
                                                   (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)):
            raise _Refused
        sides = (compare.left, compare.comparators[0])
        if not all(self._is_code(side) for side in sides):
            raise _Refused
        if any(self._code_group(side) in self.lifted_groups for side in sides):
            raise _Refused
        setattr(compare, _IDENTITY, True)

    def _code_group(self, expr):
        """The group of the functions `expr` may be, which a lifted function is one of."""
        if self._is_returner_call(expr):
            return self._call_result_group(expr)
        while isinstance(expr, ast.Subscript):
            expr = expr.value
        if expr.id in self.functions:
            return self.functions[expr.id][0]
        return self.holders.get(expr.id)

    @staticmethod
    def _is_dict_value(parent, expr):
        return isinstance(parent, ast.Dict) and any(expr is v for v in parent.values)

    def _check_value_list(self, literal):
        parent = self.parents[id(literal)]
        if isinstance(parent, (ast.List, ast.Tuple)) or self._is_dict_value(parent, literal):
            self._check_value_list(parent)
        elif not (self._is_plain_value(parent, literal) or self._is_update_value(parent, literal)):
            raise _Refused

    def _is_update_value(self, call, literal):
        """Whether `literal` is stored by a call that `_is_update` accepts."""
        return isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and \
            isinstance(call.func.value, ast.Name) and call.func.value.id in self.holders and \
            call.args[-1] is literal and self._is_update(call.func, call.func.value)

    def _is_code(self, expr):
        if isinstance(expr, ast.Name):
            return expr.id in self.functions or self.kinds.get(expr.id) == 0
        if self._is_returner_call(expr):
            return True
        return self._is_element(expr) and self._element_kind(expr) == 0

    def _check_site(self, call):
        """A call through a holder: every function of its group must take its arguments."""
        group = self.holders[self._site_holder(call)]
        self._check_arguments(call, group)
        self._check_always_evaluated(call)
        setattr(call, _SITE, group)

    def _check_arguments(self, call, group):
        """Every function of `group` takes exactly the positional arguments of `call`."""
        if call.keywords or any(isinstance(a, ast.Starred) for a in call.args):
            raise _Refused
        if not all(_takes_positional(fn, len(call.args)) for fn in self.groups[group]):
            raise _Refused

    def _site_holder(self, call):
        """The holder `call` goes through, refusing an index another scope may rebind."""
        if isinstance(call.func, ast.Name):
            return call.func.id
        node = call.func
        while isinstance(node, ast.Subscript):
            index = node.slice
            if isinstance(index, ast.Name) and any(
                    isinstance(n, (ast.Global, ast.Nonlocal)) and index.id in n.names
                    for n in ast.walk(self.module)):
                raise _Refused
            node = node.value
        return node.id

    def _check_always_evaluated(self, call):
        """Only a call evaluated whenever its statement is may be dispatched before it."""
        node, parent = call, self.parents[id(call)]
        while not isinstance(parent, ast.stmt):
            if isinstance(parent, ast.IfExp) and node is not parent.test:
                raise _Refused
            if isinstance(parent, ast.BoolOp) and node is not parent.values[0]:
                raise _Refused
            if isinstance(parent, ast.NamedExpr):
                raise _Refused
            node, parent = parent, self.parents[id(parent)]


class _Codes(ast.NodeTransformer):
    """Replace each function of a group by its code, `is` by `==` and `callable` by True."""

    def visit_Name(self, node):
        code = getattr(node, _CODE, None)
        if code is None:
            return node
        return ast.copy_location(ast.Constant(code), node)

    def visit_Compare(self, node):
        self.generic_visit(node)
        if getattr(node, _IDENTITY, False):
            node.ops = [
                ast.NotEq() if isinstance(node.ops[0], (ast.IsNot, ast.NotEq)) else ast.Eq()
            ]
        return node

    def visit_Call(self, node):
        if getattr(node, _CALLABLE, False):
            return ast.copy_location(ast.Constant(True), node)
        self.generic_visit(node)
        return node


class _Returns(ast.NodeTransformer):
    """Replace `return h(args)` by one `return f(args)` per function of h's group."""

    def __init__(self, groups):
        self.groups = groups

    def visit_Return(self, node):
        group = getattr(node, _RETURN, None)
        if group is None:
            return node
        call = node.value
        arms = [[ast.Return(ast.Call(ast.Name(fn.name, ast.Load()), copy.deepcopy(call.args), []))]
                for fn in self.groups[group]]
        return _chain(arms, call.func, node)


class _Dispatch:
    """Split the module at each dispatched call, copying the rest into one arm per function."""

    def __init__(self, groups, holders):
        self.groups = groups
        self.holders = set(holders)
        self.arms = 0

    def statements(self, stmts, before=frozenset()):
        """`stmts` with each dispatched call split, and how many copies of the end it holds.

        `before` holds the names that the enclosing code binds before `stmts`.
        """
        for i, stmt in enumerate(stmts):
            sites = [n for n in ast.walk(stmt) if getattr(n, _SITE, None) is not None]
            if not sites:
                continue
            if isinstance(stmt, _UNCOPYABLE + _COMPOUND):
                raise _Refused
            header = [stmt.test] if isinstance(stmt, ast.If) else [stmt]
            in_header = [
                n for h in header for n in ast.walk(h) if getattr(n, _SITE, None) is not None
            ]
            if isinstance(stmt, ast.If) and not in_header:
                return self._branch(stmts, i, before)
            if len(in_header) != len(sites):
                raise _Refused
            return self._split(stmts, i, min(sites, key=lambda n: (n.lineno, n.col_offset)), before)
        return stmts, 1

    def _branch(self, stmts, i, before):
        """Copy the end of `stmts` into each branch of the `if` at `i`, so a call can split."""
        stmt = stmts[i]
        before = before | _visible_names(stmts[:i] + [stmt.test])
        branches, total = [], 0
        for branch in (stmt.body, stmt.orelse):
            body, leaves = self.statements(branch + copy.deepcopy(stmts[i + 1:]), before)
            branches.append(body)
            total += leaves
        if total > _CAP:
            raise _Refused
        return stmts[:i] + [ast.copy_location(ast.If(stmt.test, *branches), stmt)], total

    def _split(self, stmts, i, site, before):
        rest = stmts[i:]
        stored = self._renamed(stmts, i, before)
        group = getattr(site, _SITE)
        callee = site.func
        setattr(site, _CHOSEN, True)
        arms, total = [], 0
        for fn in self.groups[group]:
            arm = copy.deepcopy(rest)
            _call_directly(arm[0], group, callee, fn)
            self._rename(arm, stored)
            body, leaves = self.statements(arm)
            total += leaves
            if total > _CAP:
                raise _Refused
            arms.append(body)
        delattr(site, _CHOSEN)
        return stmts[:i] + _chain(arms, callee, stmts[i]), total

    def _renamed(self, stmts, i, before):
        """The names that `stmts[i:]` bind, which each arm renames; refuse when it cannot."""
        rest = stmts[i:]
        if any(isinstance(n, _UNCOPYABLE) for s in rest for n in ast.walk(s)):
            raise _Refused
        stored = {
            n.id
            for s in rest
            for n in ast.walk(s) if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load)
        } - self.holders
        if stored & (before | _visible_names(stmts[:i])):
            raise _Refused
        return stored

    def _rename(self, arm, stored):
        self.arms += 1
        for top in arm:
            for node in ast.walk(top):
                if isinstance(node, ast.Name) and node.id in stored:
                    base = re.sub(rf"^{_PREFIX}\d+_", "", node.id)
                    node.id = f"{_PREFIX}{self.arms}_{base}"


def _hoisted_in_groups(module, path):
    """The temporaries of `_hoist_callees` that a mixed group holds."""
    candidate = copy.deepcopy(module)
    _hoist_callees(candidate)
    analysis = _propagated(candidate, path)
    roots = set(_dispatch_groups(analysis))
    return {
        key[2]
        for key in analysis.parent
        if key[0] == "name" and key[2].startswith(_HOISTED) and analysis.find(key) in roots
    }


def _check_hoisted_dispatched(module):
    """A temporary is called only through a dispatch, which calls the function directly."""
    for node in ast.walk(module):
        if isinstance(node, ast.Call) and isinstance(node.func,
                                                     ast.Name) and _HOISTED in node.func.id:
            raise _Refused


def dispatch_mixed_function_values(module, path):
    """Rewrite calls through mixed function values into dispatches, when every use allows it."""
    candidate = copy.deepcopy(module)
    lifted_methods = _may_lift_methods(module) and lift_bound_methods(candidate)
    original = _propagated(candidate if lifted_methods else module, path)
    if not _mixed_groups(original):
        return
    try:
        lifted = lift_nested_defs(candidate,
                                  _nested_members(module, original, _dispatch_groups(original)))
        _hoist_callees(candidate, _hoisted_in_groups(candidate, path))
        analysis = _propagated(candidate, path)
        plan = _Plan(candidate, analysis, _dispatch_groups(analysis), lifted)
        plan.check_uses()
        plan.check_param_holders()
        plan.check_imported()
        _Codes().visit(candidate)
        _Returns(plan.groups).visit(candidate)
        candidate.body, _ = _Dispatch(plan.groups, plan.holders).statements(candidate.body)
        _check_hoisted_dispatched(candidate)
    except _Refused:
        return
    for node in ast.walk(candidate):
        for mark in (_SITE, _CODE, _IDENTITY, _CALLABLE, _CHOSEN):
            if hasattr(node, mark):
                delattr(node, mark)
    module.body = candidate.body
    ast.fix_missing_locations(module)


def _propagated(module, path):
    analysis = _Analysis(module, path, dicts=True)
    analysis.propagate()
    return analysis
