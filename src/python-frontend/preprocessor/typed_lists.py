"""Which lists of the entry module get the typed list model (#10451).

A typed list keeps its elements in an int64 array instead of one untyped
cell per element. That is exact only for a list whose every element is an
int and whose items no code reads as untyped bytes, so this pass proves both
and leaves every other list on the current model.

The pass runs on the entry module's final AST, after every other rewrite, so
a ``for`` loop has already become an index loop over ``len`` and ``l[i]``.
Lists are tracked by abstract location: a variable of a scope (the module or
a top-level function), a function's return value or a parameter. Assigning a
name to a name, returning a name and passing a name to an analysed function
join the two locations into one class. A class is typed when it holds a
typed allocation and nothing poisons it. Anything this pass does not
recognise poisons the location involved, so any doubt keeps the current
model.

The converter lowers a node differently only when it carries
``esbmc_typed_list``. A new operation is added by a rule in ``USE_RULES``
or ``METHOD_RULES`` (and, for a new kind of allocation, a shape in
``_allocation_shape``), then a lowering of the flagged node in the converter
(python-list/typed_list.cpp).
"""

import ast
import copy

TYPED_FLAG = "esbmc_typed_list"

MODULE = "<module>"
RETURN = "<return>"

_INT_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.LShift, ast.RShift,
               ast.BitAnd, ast.BitOr, ast.BitXor)
_INT_UNARYOPS = (ast.USub, ast.UAdd, ast.Invert)
_NESTED_SCOPES = (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ListComp,
                  ast.SetComp, ast.DictComp, ast.GeneratorExp)
_INT_BUILDER = "_nondet_list_int"
_LIST_ANNOTATIONS = ("list", "List")
_VERIFIER_MODULES = ("_sv_verifier", "esbmc")
# Intrinsics whose arguments the converter lowers as ordinary expressions; a
# list used in any other __ESBMC_ call (a contract clause) is not followed.
_PLAIN_INTRINSICS = ("__ESBMC_assume", "__ESBMC_assert")
_INT64_MIN = -2**63
_INT64_MAX = 2**63 - 1


class _Classes:
    """Union-find over list locations, with the poison and allocation of each
    class kept at its root."""

    def __init__(self):
        self.parent = {}
        self.poison = {}
        self.allocated = set()

    def find(self, loc):
        self.parent.setdefault(loc, loc)
        root = loc
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[loc] != root:
            self.parent[loc], loc = root, self.parent[loc]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        self.parent[rb] = ra
        if rb in self.poison:
            self.poison.setdefault(ra, self.poison.pop(rb))
        if rb in self.allocated:
            self.allocated.discard(rb)
            self.allocated.add(ra)

    def poison_loc(self, loc, reason):
        self.poison.setdefault(self.find(loc), reason)

    def allocate(self, loc):
        self.allocated.add(self.find(loc))

    def typed(self, loc):
        root = self.find(loc)
        return root in self.allocated and root not in self.poison


def _unshare(module):
    """Copy every node an earlier rewrite placed at two positions, so each
    node has one parent."""
    seen = set()
    stack = [module]
    while stack:
        node = stack.pop()
        for field, value in ast.iter_fields(node):
            children = value if isinstance(value, list) else [value]
            for i, child in enumerate(children):
                if not isinstance(child, ast.AST):
                    continue
                if id(child) in seen:
                    child = copy.deepcopy(child)
                    if isinstance(value, list):
                        value[i] = child
                    else:
                        setattr(node, field, child)
                seen.add(id(child))
                stack.append(child)


def _parent(node):
    return getattr(node, "_typed_parent", None)


def _names_in(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _stored_names(node):
    return {
        n.id
        for n in ast.walk(node) if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load)
    }


def _walk_scope(stmts):
    """Every node of a scope's statements, not descending into a nested scope;
    the nested scope's own node is yielded."""
    stack = list(reversed(stmts))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, _NESTED_SCOPES):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def _bound_without_name_node(node):
    """Names @p node binds without an ast.Name in Store context."""
    if isinstance(node, ast.alias):
        return [node.asname or node.name.split(".")[0]]
    if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
        return [node.name]
    if isinstance(node, ast.MatchMapping) and node.rest:
        return [node.rest]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    return []


def _is_list_annotation(annotation):
    base = annotation.value if isinstance(annotation, ast.Subscript) else annotation
    return isinstance(base, ast.Name) and base.id in _LIST_ANNOTATIONS


def _allowed_list_annotation(annotation):
    if isinstance(annotation, ast.Name):
        return annotation.id in _LIST_ANNOTATIONS
    return (isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name)
            and annotation.value.id in _LIST_ANNOTATIONS and isinstance(annotation.slice, ast.Name)
            and annotation.slice.id == "int")


def _declaration(node):
    """Whether @p node only annotates a name, with a list type the typed model
    agrees with or with no list type at all."""
    return (isinstance(node, ast.AnnAssign) and node.value is None and
            (not _is_list_annotation(node.annotation) or _allowed_list_annotation(node.annotation)))


def _single_name_target(node):
    """The Name an Assign or AnnAssign binds alone, else None."""
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
        target = node.target
    else:
        return None
    return target if isinstance(target, ast.Name) else None


def _ends_in_exit(body):
    return bool(body) and isinstance(body[-1], (ast.Return, ast.Raise))


class _Scope:
    """A module or a top-level function: its statements and which names are
    its own."""

    def __init__(self, name, body, params=()):
        self.name = name
        self.body = body
        self.params = list(params)
        self.locals = set(self.params)
        declared = set()
        for node in _walk_scope(body):
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                declared.update(node.names)
            elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                self.locals.add(node.id)
            else:
                self.locals.update(_bound_without_name_node(node))
        self.declared = declared
        self.locals -= declared

    def loc(self, name):
        if self.name == MODULE or name in self.locals:
            return (self.name, name)
        return (MODULE, name)


# A use rule's answer for a use that is not its shape.
NOT_MINE = object()


class TypedListGate:
    """Decide the typed lists of one module and flag their nodes."""

    def __init__(self, module):
        self.module = module
        self.classes = _Classes()
        self.scopes = []
        self.functions = {}
        self.bindings = {}
        self.int_vars = {}
        self.shadowed = set()
        self.verifier_modules = set()
        # Sites recorded once and checked again each round.
        self.allocations = []  # (scope, value node, location)
        self.accesses = []  # (scope, Subscript, stored value or None)
        self.appends = []  # (scope, Call)

    # --- scopes and the functions whose calls carry lists -------------------

    def _build_scopes(self):
        defs = {}
        for stmt in self.module.body:
            if isinstance(stmt, ast.FunctionDef):
                defs.setdefault(stmt.name, []).append(stmt)
        module_scope = _Scope(MODULE, self.module.body)
        self.scopes = [module_scope]
        # A name defined twice is not followed: its bodies count as nested.
        self.top_defs = {name: found[0] for name, found in defs.items() if len(found) == 1}
        for func in self.top_defs.values():
            self.scopes.append(_Scope(func.name, func.body, [a.arg for a in func.args.args]))
        for scope in self.scopes:
            self.shadowed |= scope.locals
        self.functions = self._analysed_functions()

    def _walked_calls(self):
        calls = set()
        for scope in self.scopes:
            calls.update(id(n) for n in _walk_scope(scope.body) if isinstance(n, ast.Call))
        return calls

    def _analysed_functions(self):
        """Top-level functions whose every use is a plain positional call from
        code this pass walks, so a list passed to one reaches its parameter
        and nothing else does."""
        candidates = {
            name: func
            for name, func in self.top_defs.items() if self._plain_signature(func)
        }
        walked = self._walked_calls()
        for node in ast.walk(self.module):
            for name in self._disqualified(node, candidates, walked):
                candidates.pop(name, None)
        return candidates

    def _disqualified(self, node, candidates, walked):
        """The candidate functions @p node uses in a way not followed."""
        if isinstance(node, ast.Name) and node.id in candidates:
            call = _parent(node)
            followed = (isinstance(node.ctx, ast.Load) and isinstance(call, ast.Call)
                        and call.func is node and id(call) in walked
                        and self._plain_call(call, candidates[node.id]))
            return [] if followed else [node.id]
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return node.names
        return [
            name for name in _bound_without_name_node(node)
            if name in candidates and node is not candidates[name]
        ]

    @staticmethod
    def _plain_signature(func):
        args = func.args
        if func.decorator_list or args.posonlyargs or args.vararg or args.kwonlyargs:
            return False
        if args.kwarg or args.defaults:
            return False
        return not any(
            isinstance(n, (ast.Yield, ast.YieldFrom, ast.Await)) for n in _walk_scope(func.body))

    @staticmethod
    def _plain_call(call, func):
        return (not call.keywords and len(call.args) == len(func.args.args)
                and not any(isinstance(a, ast.Starred) for a in call.args))

    # --- int-valued expressions --------------------------------------------

    def is_int(self, expr, scope):
        """Whether @p expr always evaluates to an int (never a bool)."""
        if isinstance(expr, ast.Constant):
            return isinstance(
                expr.value,
                int) and not isinstance(expr.value, bool) and _INT64_MIN <= expr.value <= _INT64_MAX
        if isinstance(expr, ast.Name):
            return self.int_vars.get(scope.loc(expr.id), False)
        if isinstance(expr, ast.BinOp):
            return isinstance(expr.op, _INT_BINOPS) and self.is_int(
                expr.left, scope) and self.is_int(expr.right, scope)
        if isinstance(expr, ast.UnaryOp):
            return isinstance(expr.op, _INT_UNARYOPS) and self.is_int(expr.operand, scope)
        if isinstance(expr, ast.IfExp):
            return self.is_int(expr.body, scope) and self.is_int(expr.orelse, scope)
        if isinstance(expr, ast.Call):
            return self._is_int_call(expr)
        if isinstance(expr, ast.Subscript):
            return (getattr(expr, "_typed_access", False)
                    and self.classes.typed(scope.loc(expr.value.id)))
        return False

    def _is_int_call(self, call):
        if call.keywords:
            return False
        if isinstance(call.func, ast.Attribute):
            return (call.func.attr == "nondet_int" and not call.args
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id in self.verifier_modules)
        if not isinstance(call.func, ast.Name):
            return False
        if call.func.id in self.functions:
            return self.int_vars.get((call.func.id, RETURN), False)
        if call.func.id in self.shadowed:
            return False
        if call.func.id == "len":
            return len(call.args) == 1 and not isinstance(call.args[0], ast.Starred)
        return call.func.id == "nondet_int" and not call.args

    def _verifier_modules(self):
        """Names bound only by `import _sv_verifier` or `import esbmc`."""
        imported = {
            alias.asname or alias.name
            for node in ast.walk(self.module) if isinstance(node, ast.Import)
            for alias in node.names if alias.name in _VERIFIER_MODULES
        }
        other = set()
        for node in ast.walk(self.module):
            if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                other.add(node.id)
            elif not isinstance(node, ast.alias) or not isinstance(_parent(node), ast.Import):
                other.update(_bound_without_name_node(node))
            elif node.name not in _VERIFIER_MODULES:
                other.add(node.asname or node.name.split(".")[0])
        return imported - other

    def _collect_bindings(self):
        """Every value each variable is bound to; None for a binding whose
        value this pass does not follow."""
        for scope in self.scopes:
            if scope.name not in self.functions:
                for param in scope.params:
                    self._bind(scope, param, None)
            elif not _ends_in_exit(scope.body):
                self._bind_location((scope.name, RETURN), None, scope)
            for node in _walk_scope(scope.body):
                self._collect_flow_bindings(node, scope)
                self._collect_name_bindings(node, scope)

    def _collect_flow_bindings(self, node, scope):
        """A followed function's return values and its arguments at a call."""
        if isinstance(node, ast.Return) and scope.name in self.functions:
            self._bind_location((scope.name, RETURN), node.value, scope)
        if self._analysed_call(node):
            for arg, param in zip(node.args, self.functions[node.func.id].args.args):
                self._bind_location((node.func.id, param.arg), arg, scope)

    def _collect_name_bindings(self, node, scope):
        if isinstance(node, _NESTED_SCOPES) and node not in self.top_defs.values():
            # A walrus in a comprehension, or a nonlocal or global store in a
            # nested function, rebinds a variable of this scope or the module.
            for name in _stored_names(node):
                self._bind(scope, name, None)
                self._bind_location((MODULE, name), None, scope)
            return
        target = _single_name_target(node)
        if target is not None:
            self._bind(scope, target.id, node.value)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            load = ast.Name(id=node.target.id, ctx=ast.Load())
            self._bind(scope, node.target.id, ast.BinOp(left=load, op=node.op, right=node.value))
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            parent = _parent(node)
            if _single_name_target(parent) is not node and not isinstance(parent, ast.AugAssign):
                self._bind(scope, node.id, None)
        else:
            for name in _bound_without_name_node(node):
                self._bind(scope, name, None)

    def _bind_location(self, loc, value, scope):
        self.bindings.setdefault(loc, []).append((value, scope))

    def _bind(self, scope, name, value):
        self._bind_location(scope.loc(name), value, scope)

    def _compute_int_vars(self):
        """Greatest fixpoint: a variable is int when every binding of it stores
        an int, given the variables currently assumed int."""
        self.int_vars = dict.fromkeys(self.bindings, True)
        changed = True
        while changed:
            changed = False
            for loc, values in self.bindings.items():
                if self.int_vars[loc] and not all(value is not None and self.is_int(value, scope)
                                                  for value, scope in values):
                    self.int_vars[loc] = False
                    changed = True

    # --- recording sites ---------------------------------------------------

    def _allocation_shape(self, value):
        """Whether @p value builds a fresh list the typed model could hold;
        its elements are checked later."""
        if isinstance(value, ast.List):
            return not any(isinstance(e, ast.Starred) for e in value.elts)
        return (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id == _INT_BUILDER and _INT_BUILDER not in self.shadowed
                and len(value.args) == 1 and not value.keywords
                and not isinstance(value.args[0], ast.Starred))

    def _flow_into(self, loc, value, scope):
        """Record that @p value is stored to location @p loc."""
        if self._allocation_shape(value):
            self.allocations.append((scope, value, loc))
            self.classes.allocate(loc)
        elif isinstance(value, ast.Name):
            self.classes.union(loc, scope.loc(value.id))
        elif self._analysed_call(value):
            self.classes.union(loc, (value.func.id, RETURN))
        else:
            self.classes.poison_loc(loc, f"holds {_describe(value)}")

    def _analysed_call(self, value):
        return (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id in self.functions)

    def _record_scope(self, scope):
        if scope.name != MODULE:
            self._record_function_edges(scope)
        for node in _walk_scope(scope.body):
            if isinstance(node, _NESTED_SCOPES):
                self._record_nested(node, scope)
            elif _single_name_target(node) is not None:
                self._record_assignment(node, scope)
            elif isinstance(node, ast.Return):
                self._record_return(node, scope)
            elif isinstance(node, ast.Call):
                self._record_call(node, scope)
            elif isinstance(node, ast.Name):
                self._record_name(node, scope)
        for name in scope.declared:
            self.classes.poison_loc(scope.loc(name), "declared global or nonlocal")
            self.classes.poison_loc((MODULE, name), "declared global or nonlocal")

    def _record_function_edges(self, scope):
        ret = (scope.name, RETURN)
        if scope.name not in self.functions:
            self.classes.poison_loc(ret, "returned by a function not followed")
            for param in scope.params:
                self.classes.poison_loc((scope.name, param), "parameter of a function not followed")
        elif not _ends_in_exit(scope.body):
            self.classes.poison_loc(ret, "a function that may return None")
        func = self.top_defs[scope.name]
        if func.returns is not None and _is_list_annotation(
                func.returns) and not _allowed_list_annotation(func.returns):
            self.classes.poison_loc(ret, "annotated with another list type")
        for arg in func.args.args:
            if arg.annotation is not None and _is_list_annotation(
                    arg.annotation) and not _allowed_list_annotation(arg.annotation):
                self.classes.poison_loc((scope.name, arg.arg), "annotated with another list type")

    def _record_nested(self, node, scope):
        if scope.name == MODULE and node in self.top_defs.values():
            # A followed function is its own scope; only what the definition
            # evaluates here is a use.
            parts = node.decorator_list + node.args.defaults + node.args.kw_defaults
            names = set().union(*(_names_in(p) for p in parts if p is not None))
        else:
            names = _names_in(node)
        reason = f"used in a nested {type(node).__name__}"
        for name in names:
            self.classes.poison_loc(scope.loc(name), reason)
            self.classes.poison_loc((MODULE, name), reason)

    def _record_assignment(self, node, scope):
        target = _single_name_target(node)
        loc = scope.loc(target.id)
        if isinstance(node, ast.AnnAssign) and _is_list_annotation(
                node.annotation) and not _allowed_list_annotation(node.annotation):
            self.classes.poison_loc(loc, "annotated with another list type")
        self._flow_into(loc, node.value, scope)

    def _record_return(self, node, scope):
        if scope.name == MODULE:
            return
        loc = (scope.name, RETURN)
        if node.value is None:
            self.classes.poison_loc(loc, "a function that may return None")
        else:
            self._flow_into(loc, node.value, scope)

    def _record_call(self, node, scope):
        if self._analysed_call(node):
            params = self.functions[node.func.id].args.args
            for arg, param in zip(node.args, params):
                self._flow_into((node.func.id, param.arg), arg, scope)
            if not self._result_followed(node, scope):
                self.classes.poison_loc((node.func.id, RETURN),
                                        f"result used in {_describe(_parent(node))}")
        elif isinstance(node.func, ast.Name) and node.func.id.startswith(
                "__ESBMC_") and node.func.id not in _PLAIN_INTRINSICS:
            for name in _names_in(node):
                self.classes.poison_loc(scope.loc(name), f"used in {node.func.id}")

    def _result_followed(self, call, scope):
        """Whether the list an analysed call returns reaches only a location
        `_flow_into` joins it to, or is dropped."""
        parent = _parent(call)
        if isinstance(parent, ast.Expr) or (_single_name_target(parent) is not None
                                            and parent.value is call):
            return True
        if isinstance(parent, ast.Return):
            return scope.name != MODULE
        return isinstance(parent, ast.Call) and call in parent.args and self._analysed_call(parent)

    def _record_name(self, node, scope):
        loc = scope.loc(node.id)
        if not isinstance(node.ctx, ast.Load):
            parent = _parent(node)
            if _single_name_target(parent) is not node and not _declaration(parent):
                self.classes.poison_loc(loc, f"bound by {_describe(parent)}")
            return
        reason = self._classify_use(node, scope)
        if reason is not None:
            self.classes.poison_loc(loc, reason)

    def _classify_use(self, node, scope):
        """Record the use of name @p node; the poison reason when the use is
        not one the typed model implements, else None."""
        for rule in self.USE_RULES:
            verdict = rule(self, node, scope)
            if verdict is not NOT_MINE:
                return verdict
        return f"used in {_describe(_parent(node))}"

    # Use rules. Each looks at a load of a list name and returns NOT_MINE when
    # the use is not its shape, None when the typed model implements it, or a
    # poison reason. A rule that accepts a use must also record whatever the
    # converter needs flagged (see `accesses` and `appends`) and, for a value
    # stored into the list, have it checked to be an int in `_check_values`.

    def _rule_len(self, node, _scope):
        parent = _parent(node)
        if not (isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name)
                and parent.func.id == "len" and parent.args == [node]):
            return NOT_MINE
        return "passed to a shadowed len" if "len" in self.shadowed else None

    def _rule_analysed_call(self, node, _scope):
        parent = _parent(node)
        if isinstance(parent, ast.Call) and node in parent.args and self._analysed_call(parent):
            return None
        return NOT_MINE

    def _rule_alias(self, node, scope):
        """`y = x` and `return x`: the locations were joined when recorded."""
        parent = _parent(node)
        if _single_name_target(parent) is not None and parent.value is node:
            return None
        if isinstance(parent, ast.Return) and scope.name != MODULE:
            return None
        return NOT_MINE

    def _rule_subscript(self, node, scope):
        """`x[i]`, `x[i] = v` and the `x[i] = x[i] op v` an augmented
        assignment becomes."""
        subscript = _parent(node)
        if not (isinstance(subscript, ast.Subscript) and subscript.value is node):
            return NOT_MINE
        if isinstance(subscript.slice, ast.Slice):
            return "sliced"
        holder = _parent(subscript)
        if isinstance(subscript.ctx, ast.Store):
            if not (isinstance(holder, ast.Assign) and holder.targets == [subscript]):
                return f"element bound by {_describe(holder)}"
            value = holder.value
        elif isinstance(subscript.ctx, ast.Load):
            value = None
        else:
            return "element deleted"
        setattr(subscript, "_typed_access", True)
        self.accesses.append((scope, subscript, value))
        return None

    def _rule_method(self, node, scope):
        attribute = _parent(node)
        if not (isinstance(attribute, ast.Attribute) and attribute.value is node):
            return NOT_MINE
        call = _parent(attribute)
        method = self.METHOD_RULES.get(attribute.attr)
        if method is None or not (isinstance(call, ast.Call) and call.func is attribute):
            return f"method .{attribute.attr}"
        return method(self, call, scope)

    # Method rules, by name: @p call is `x.<name>(...)` on a list name.

    def _method_append(self, call, scope):
        if not (len(call.args) == 1 and not call.keywords and
                not isinstance(call.args[0], ast.Starred) and isinstance(_parent(call), ast.Expr)):
            return "method .append used as a value"
        self.appends.append((scope, call))
        return None

    # The uses of a list name the typed model implements, tried in order.
    USE_RULES = (_rule_len, _rule_analysed_call, _rule_alias, _rule_subscript, _rule_method)

    # The list methods it implements.
    METHOD_RULES = {"append": _method_append}

    # --- the decision -------------------------------------------------------

    def _check_values(self):
        """Poison every class that a recorded site indexes or fills with a
        non-int. True when anything changed."""
        changed = False

        def poison(loc, reason):
            nonlocal changed
            if self.classes.typed(loc):
                self.classes.poison_loc(loc, reason)
                changed = True

        for scope, value, loc in self.allocations:
            if isinstance(value, ast.List) and not all(self.is_int(e, scope) for e in value.elts):
                poison(loc, "built from a non-int")
            elif isinstance(value, ast.Call) and not self.is_int(value.args[0], scope):
                poison(loc, "built with a non-int size")
        for scope, subscript, value in self.accesses:
            loc = scope.loc(subscript.value.id)
            if not self.is_int(subscript.slice, scope):
                poison(loc, "indexed by a non-int " + ast.unparse(subscript.slice))
            elif value is not None and not self.is_int(value, scope):
                poison(loc, "stores a non-int element")
        for scope, call in self.appends:
            if not self.is_int(call.args[0], scope):
                poison(scope.loc(call.func.value.id), "appends a non-int")
        return changed

    def run(self):
        _unshare(self.module)
        for node in ast.walk(self.module):
            for child in ast.iter_child_nodes(node):
                setattr(child, "_typed_parent", node)
        self.verifier_modules = self._verifier_modules()
        self._build_scopes()
        for scope in self.scopes:
            self._record_scope(scope)
        self._collect_bindings()
        while True:
            self._compute_int_vars()
            if not self._check_values():
                break
        self._flag()
        for node in ast.walk(self.module):
            node.__dict__.pop("_typed_parent", None)
            node.__dict__.pop("_typed_access", None)
        return self

    def _flag(self):
        for _, value, loc in self.allocations:
            if self.classes.typed(loc):
                setattr(value, TYPED_FLAG, True)
        for scope, subscript, _ in self.accesses:
            if self.classes.typed(scope.loc(subscript.value.id)):
                setattr(subscript, TYPED_FLAG, True)
        for scope, call in self.appends:
            if self.classes.typed(scope.loc(call.func.value.id)):
                setattr(call, TYPED_FLAG, True)

    def builder_sites(self):
        """(typed, poison reason) for each nondet_list builder site."""
        return [(self.classes.typed(loc), self.classes.poison.get(self.classes.find(loc)))
                for _, value, loc in self.allocations if isinstance(value, ast.Call)]


def _describe(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_describe(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return f"a call to {_describe(node.func)}"
    if isinstance(node, ast.Compare):
        return "a comparison " + "/".join(type(op).__name__ for op in node.ops)
    return type(node).__name__ if node is not None else "nothing"


def mark_typed_lists(module):
    """Flag the nodes of @p module's typed lists; returns the gate."""
    return TypedListGate(module).run()
