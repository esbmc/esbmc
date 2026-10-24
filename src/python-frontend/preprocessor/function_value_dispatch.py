"""Dispatch a call through a function value whose functions return different types (#8970).

The converter gives a call through a function value one return type, so
function_value_types refuses such a call when its functions disagree. For a
group of module-level functions whose values only flow through module-level
names, lists and tuples, this pass rewrites the program so that no function
value is left:

- each reference to a function of the group becomes its index in the group,
  so `h = f if c else g` becomes `h = 0 if c else 1`
- `h is f` becomes `h == 0` and `callable(h)` becomes `True`
- a statement calling `h()` becomes `if h == 0: ... elif ...: ... else: ...`,
  and each arm holds that statement and the rest of the module, with `h()`
  replaced by a direct call to that arm's function.

Each arm then sees the precise type of its call's result. A name bound in an
arm is renamed per arm, so that it does not join values of different types.
When any condition does not hold, the module is left as it was and
function_value_types refuses the call as before.
"""

import ast
import copy
import re

from preprocessor.function_value_types import _Analysis, _differ, _own_nodes

# The most copies of the rest of the module that one module may need.
_CAP = 8
_PREFIX = "_esbmc_fv"
# Marks left on nodes between checking a use and rewriting it.
_SITE = "esbmc_fv_site"
_CODE = "esbmc_fv_code"
_IDENTITY = "esbmc_fv_identity"
_CALLABLE = "esbmc_fv_callable"
_CHOSEN = "esbmc_fv_chosen"
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


def _sole_def(fn, defs, module, analysis):
    """Whether `fn` is the module's only, undecorated def of its name, never rebound."""
    return (defs.get(getattr(fn, "name", None)) is fn and not fn.decorator_list
            and len(analysis.defs.get((id(module), fn.name), [])) == 1 and not any(
                isinstance(n, ast.Name) and n.id == fn.name and isinstance(n.ctx, ast.Store)
                for n in _own_nodes(module)))


def _takes_positional(fn, count):
    """Whether `fn` takes exactly `count` positional arguments and nothing else."""
    args = fn.args
    return not (args.vararg or args.kwarg or args.kwonlyargs or args.defaults
                or args.posonlyargs) and len(args.args) == count


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

    def __init__(self, module, analysis, roots):
        self.module = module
        self.parents = _parents(module)
        self.functions = {}
        self.groups = []
        self.holders = {}
        defs = {stmt.name: stmt for stmt in module.body if isinstance(stmt, ast.FunctionDef)}
        for root in roots:
            group = len(self.groups)
            fns = sorted((m[0] for m in analysis.members[root].values()), key=lambda f: f.lineno)
            if not all(_sole_def(fn, defs, module, analysis) for fn in fns):
                raise _Refused
            self.groups.append(fns)
            for code, fn in enumerate(fns):
                self.functions[fn.name] = (group, code)
            self._add_holders(analysis, root, group)
        self.kinds = {}
        self.values = {}
        self._collect_bindings()
        for name in self.holders:
            self._kind(name, set())

    def _add_holders(self, analysis, root, group):
        """Record the module names in `root`'s group; refuse any other kind of value in it."""
        for key in [k for k in analysis.parent if analysis.find(k) == root]:
            if key[0] == "elements":
                continue
            if key[0] != "name" or key[1] != id(self.module):
                raise _Refused
            if key[2] not in self.functions:
                self.holders[key[2]] = group

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
        """'function' or 'list': what holder `name` holds."""
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
                return "function"
            if expr.id in self.holders:
                return self._kind(expr.id, seen)
        elif isinstance(expr, ast.IfExp):
            kinds = {self._value_kind(expr.body, seen), self._value_kind(expr.orelse, seen)}
            if len(kinds) == 1:
                return kinds.pop()
        elif isinstance(expr, (ast.List, ast.Tuple)):
            if expr.elts and all(self._value_kind(e, seen) == "function" for e in expr.elts):
                return "list"
        elif self._is_element(expr):
            if self._value_kind(expr.value, seen) == "list":
                return "function"
        raise _Refused

    @staticmethod
    def _is_element(expr):
        """`xs[i]` with a name or literal index, whose evaluation has no effects."""
        if not isinstance(expr, ast.Subscript) or not isinstance(expr.value, ast.Name):
            return False
        index = expr.slice
        if isinstance(index, ast.UnaryOp) and isinstance(index.op, ast.USub):
            index = index.operand
        return isinstance(index, ast.Name) or (isinstance(index, ast.Constant)
                                               and isinstance(index.value, int))

    # uses
    def check_uses(self):
        names = set(self.functions) | set(self.holders)
        bound = _bound_names([self.module])
        if bound & _REFLECTIVE or any(n.startswith(_PREFIX) for n in bound):
            raise _Refused
        for node in ast.walk(self.module):
            if isinstance(node, ast.Attribute) and node.attr in _REFLECTIVE_ATTRS:
                raise _Refused
            if isinstance(node, ast.Name):
                if node.id in names:
                    self._check_use(node)
            elif node is not self.module and not isinstance(node, ast.FunctionDef) and \
                    _bound_names([node]) & names and _binds_by_string(node, names):
                raise _Refused
            elif isinstance(node, ast.FunctionDef) and node.name in self.holders:
                raise _Refused

    def _nested(self, node):
        parent = self.parents.get(id(node))
        while parent is not None:
            if isinstance(parent, _NESTED):
                return True
            parent = self.parents.get(id(parent))
        return False

    def _check_use(self, node):
        parent = self.parents[id(node)]
        if _is_callee(parent, node) and node.id in self.functions:
            return
        if self._nested(node):
            raise _Refused
        if isinstance(node.ctx, ast.Store):
            self._check_store(node)
        elif isinstance(node.ctx, ast.Del):
            raise _Refused
        elif node.id in self.functions:
            self._check_value(node)
            setattr(node, _CODE, self.functions[node.id][1])
        elif self.kinds[node.id] == "function":
            self._check_function_holder(node, parent)
        else:
            self._check_list_holder(node, parent)

    def _check_store(self, node):
        if node.id in self.functions or not self._assign_target(node):
            raise _Refused

    def _check_function_holder(self, node, parent):
        if _is_callee(parent, node):
            self._check_site(parent)
        else:
            self._check_value(node)

    def _check_list_holder(self, node, parent):
        if isinstance(parent, ast.Subscript) and parent.value is node:
            if not self._is_element(parent) or not isinstance(parent.ctx, ast.Load):
                raise _Refused
            outer = self.parents[id(parent)]
            if _is_callee(outer, parent):
                self._check_site(outer)
            else:
                self._check_value(parent)
        elif not (_is_builtin_call(parent, "len", node) or self._is_plain_value(parent, node)):
            raise _Refused

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
        """Whether every target of `assign` is a name or a tuple of names."""
        return all(
            isinstance(n, (ast.Name, ast.Tuple, ast.List)) for t in assign.targets
            for n in ast.walk(t) if not isinstance(n, (ast.expr_context)))

    def _check_value(self, expr):
        """`expr`, a function or an element, flows only where a code can stand for it."""
        parent = self.parents[id(expr)]
        if isinstance(parent, ast.IfExp) and expr is not parent.test:
            self._check_value(parent)
        elif isinstance(parent, (ast.List, ast.Tuple)):
            self._check_value_list(parent)
        elif self._is_plain_value(parent, expr):
            pass
        elif isinstance(parent, ast.Compare):
            self._check_identity(parent)
        elif _is_builtin_call(parent, "callable", expr) and not _rebinds(self.module, "callable"):
            setattr(parent, _CALLABLE, True)
        else:
            raise _Refused

    def _is_plain_value(self, parent, expr):
        """Whether `expr` is the whole value of an assignment to names."""
        return isinstance(parent, ast.Assign) and parent.value is expr and \
            self._plain_targets(parent)

    def _check_identity(self, compare):
        """`compare` tests two codes for identity or equality."""
        if len(compare.ops) != 1 or not isinstance(compare.ops[0],
                                                   (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)):
            raise _Refused
        if not all(self._is_code(side) for side in (compare.left, compare.comparators[0])):
            raise _Refused
        setattr(compare, _IDENTITY, True)

    def _check_value_list(self, literal):
        if not self._is_plain_value(self.parents[id(literal)], literal):
            raise _Refused

    def _is_code(self, expr):
        if isinstance(expr, ast.Name):
            return expr.id in self.functions or self.kinds.get(expr.id) == "function"
        return self._is_element(expr) and self.kinds.get(expr.value.id) == "list"

    def _check_site(self, call):
        """A call through a holder: every function of its group must take its arguments."""
        if call.keywords or any(isinstance(a, ast.Starred) for a in call.args):
            raise _Refused
        group = self.holders[self._site_holder(call)]
        if not all(_takes_positional(fn, len(call.args)) for fn in self.groups[group]):
            raise _Refused
        self._check_always_evaluated(call)
        setattr(call, _SITE, group)

    def _site_holder(self, call):
        """The holder `call` goes through, refusing an index another scope may rebind."""
        if isinstance(call.func, ast.Name):
            return call.func.id
        index = call.func.slice
        if isinstance(index, ast.Name) and any(
                isinstance(n, (ast.Global, ast.Nonlocal)) and index.id in n.names
                for n in ast.walk(self.module)):
            raise _Refused
        return call.func.value.id

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


class _Dispatch:
    """Split the module at each dispatched call, copying the rest into one arm per function."""

    def __init__(self, groups, holders):
        self.groups = groups
        self.holders = set(holders)
        self.arms = 0

    def statements(self, stmts):
        """`stmts` with each dispatched call split, and how many copies of the end it holds."""
        for i, stmt in enumerate(stmts):
            sites = [n for n in ast.walk(stmt) if getattr(n, _SITE, None) is not None]
            if not sites:
                continue
            header = [stmt.test] if isinstance(stmt, ast.If) else [stmt]
            if isinstance(stmt, _UNCOPYABLE + _COMPOUND):
                raise _Refused
            if len([n for h in header for n in ast.walk(h)
                    if getattr(n, _SITE, None) is not None]) != len(sites):
                raise _Refused
            return self._split(stmts, i, min(sites, key=lambda n: (n.lineno, n.col_offset)))
        return stmts, 1

    def _split(self, stmts, i, site):
        rest = stmts[i:]
        stored = self._renamed(stmts, i)
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

    def _renamed(self, stmts, i):
        """The names that `stmts[i:]` bind, which each arm renames; refuse when it cannot."""
        rest = stmts[i:]
        if any(isinstance(n, _UNCOPYABLE) for s in rest for n in ast.walk(s)):
            raise _Refused
        stored = {
            n.id
            for s in rest
            for n in ast.walk(s) if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load)
        } - self.holders
        if stored & _bound_names(stmts[:i]):
            raise _Refused
        return stored

    def _rename(self, arm, stored):
        self.arms += 1
        for top in arm:
            for node in ast.walk(top):
                if isinstance(node, ast.Name) and node.id in stored:
                    base = re.sub(rf"^{_PREFIX}\d+_", "", node.id)
                    node.id = f"{_PREFIX}{self.arms}_{base}"


def dispatch_mixed_function_values(module, path):
    """Rewrite calls through mixed function values into dispatches, when every use allows it."""
    if not _mixed_groups(_propagated(module, path)):
        return
    candidate = copy.deepcopy(module)
    analysis = _propagated(candidate, path)
    try:
        plan = _Plan(candidate, analysis, _mixed_groups(analysis))
        plan.check_uses()
        _Codes().visit(candidate)
        candidate.body, _ = _Dispatch(plan.groups, plan.holders).statements(candidate.body)
    except _Refused:
        return
    for node in ast.walk(candidate):
        for mark in (_SITE, _CODE, _IDENTITY, _CALLABLE, _CHOSEN):
            if hasattr(node, mark):
                delattr(node, mark)
    module.body = candidate.body
    ast.fix_missing_locations(module)


def _propagated(module, path):
    analysis = _Analysis(module, path)
    analysis.propagate()
    return analysis
