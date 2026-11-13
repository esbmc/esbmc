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
# On a typed node of a list whose elements are bools, held as 0 or 1.
BOOL_FLAG = "esbmc_typed_bool"
# On a `sorted(x)` call of a typed x whose result is only compared by `==` or
# `!=` with a list: the converter sorts x and copies it into an untyped list.
UNTYPED_RESULT_FLAG = "esbmc_typed_list_source"
# On a slice, list() or copy() of a typed list, whose result may be untyped.
SOURCE_FLAG = "esbmc_typed_source"

MODULE = "<module>"
RETURN = "<return>"

_INT_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.LShift, ast.RShift,
               ast.BitAnd, ast.BitOr, ast.BitXor)
_INT_UNARYOPS = (ast.USub, ast.UAdd, ast.Invert)
_NESTED_SCOPES = (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ListComp,
                  ast.SetComp, ast.DictComp, ast.GeneratorExp)
_INT_BUILDER = "_nondet_list_int"
_TRUTH_TESTS = (ast.If, ast.While, ast.IfExp, ast.Assert)
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
            and annotation.slice.id in ("int", "bool", "Any"))


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


def _literal_step(step):
    """Whether a slice's step is absent or an int literal, maybe negated, that
    fits an int64 with its negation."""
    if isinstance(step, ast.UnaryOp) and isinstance(step.op, ast.USub):
        step = step.operand
    if step is None:
        return True
    value = getattr(step, "value", None) if isinstance(step, ast.Constant) else None
    return isinstance(value, int) and not isinstance(value, bool) and value <= _INT64_MAX


def _ends_in_exit(body):
    return bool(body) and isinstance(body[-1], (ast.Return, ast.Raise))


def _is_none(node):
    return isinstance(node, ast.Constant) and node.value is None


def _stored_value(target):
    """The value `x[i] = v` or `x[i], y = v, w` stores into @p target, else
    None."""
    holder = _parent(target)
    if isinstance(holder, ast.Assign) and holder.targets == [target]:
        return holder.value
    return _unpacked_value(target)


def _none_tested(node):
    """Whether @p node is compared only by `is None` or `is not None`."""
    compare = _parent(node)
    if not (isinstance(compare, ast.Compare) and len(compare.ops) == 1
            and isinstance(compare.ops[0], (ast.Is, ast.IsNot))):
        return False
    return _is_none(compare.comparators[0] if compare.left is node else compare.left)


def _unpacked_value(target):
    """The value `a[i], b = x, y` stores into its target @p target, paired
    by position as the converter's desugaring pairs them, else None."""
    holder = _parent(target)
    assign = _parent(holder)
    if not (isinstance(holder, (ast.Tuple, ast.List)) and isinstance(assign, ast.Assign)
            and assign.targets == [holder] and isinstance(assign.value, (ast.Tuple, ast.List))
            and len(assign.value.elts) == len(holder.elts)):
        return None
    if not all(isinstance(t, (ast.Name, ast.Subscript, ast.Attribute)) for t in holder.elts):
        return None
    if any(isinstance(v, ast.Starred) for v in assign.value.elts):
        return None
    return assign.value.elts[[id(t) for t in holder.elts].index(id(target))]


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
        self.bool_vars = {}
        self.bool_roots = set()  # classes whose elements are bools
        self.annotated = []  # (location, whether its list annotation says bool)
        self.dict_stores = {}  # dict location: [(stored value, scope)]
        self.int_dicts = {}
        self.shadowed = set()
        self.verifier_modules = set()
        # Sites recorded once and checked again each round.
        self.allocations = []  # (scope, value node, location)
        self.accesses = []  # (scope, Subscript, stored value or None)
        self.appends = []  # (scope, Call)
        self.calls = []  # calls of followed functions
        self.sorts = []  # (scope, Call)
        self.copies = []  # (scope, slice/list()/copy() node, source Name)
        self.compares = []  # (scope, Compare)
        self.extends = []  # (scope, Call)
        self.slice_stores = []  # (scope, Subscript stored by a slice)

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
            return self._is_int_subscript(expr, scope)
        return False

    def _is_int_subscript(self, expr, scope):
        """An item of a typed list or of a dict whose items are ints."""
        if self._typed_read(expr, scope):
            return not self._holds_bools(scope.loc(_list_name(expr.value).id))
        return (isinstance(expr.value, ast.Name) and not isinstance(expr.slice, ast.Slice)
                and self.int_dicts.get(scope.loc(expr.value.id), False))

    def _typed_read(self, expr, scope):
        return getattr(expr, "_typed_access", False) and self.classes.typed(
            scope.loc(_list_name(expr.value).id))

    def _holds_bools(self, loc):
        return self.classes.find(loc) in self.bool_roots

    def is_bool(self, expr, scope):
        """Whether @p expr always evaluates to a bool."""
        if isinstance(expr, ast.Constant):
            return isinstance(expr.value, bool)
        if isinstance(expr, ast.UnaryOp):
            return isinstance(expr.op, ast.Not)
        if isinstance(expr, ast.Compare):
            # Ints and bools compare to a bool; other operands may not.
            return all(
                self.is_int(o, scope) or self.is_bool(o, scope)
                for o in [expr.left] + expr.comparators)
        if isinstance(expr, ast.BoolOp):
            return all(self.is_bool(v, scope) for v in expr.values)
        if isinstance(expr, ast.IfExp):
            return self.is_bool(expr.body, scope) and self.is_bool(expr.orelse, scope)
        if isinstance(expr, ast.Name):
            return self.bool_vars.get(scope.loc(expr.id), False)
        if isinstance(expr, ast.Subscript):
            return self._typed_read(expr, scope) and self._holds_bools(
                scope.loc(_list_name(expr.value).id))
        return False

    def is_element(self, expr, scope, loc):
        """Whether @p expr is an element of the kind list @p loc holds."""
        if self._holds_bools(loc):
            return self.is_bool(expr, scope)
        return self.is_int(expr, scope)

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
                self._bind_location((scope.name, RETURN), ast.Constant(value=None), scope)
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

    def _poison_unfollowed_bindings(self):
        """A location bound by an import, a handler or any other binding this
        pass does not follow, or never bound in this module, may hold
        anything."""
        for loc in list(self.classes.parent):
            values = self.bindings.get(loc)
            if not values or any(value is None for value, _ in values):
                self.classes.poison_loc(loc, "bound by a binding not followed")

    def _bind(self, scope, name, value):
        self._bind_location(scope.loc(name), value, scope)

    def _collect_dict_stores(self):
        """The values stored into each dict variable bound only to dict
        displays and used only to read or store one item by subscript. Any
        other binding or use, including one in a nested scope, leaves the
        dict out."""
        stores = {
            loc: [(v, scope) for display, scope in values for v in display.values]
            for loc, values in self.bindings.items() if loc[1] != RETURN and all(
                isinstance(v, ast.Dict) and None not in v.keys for v, _ in values)
        }
        escaped = set()
        for scope in self.scopes:
            escaped |= self._record_dict_uses(scope, stores)
        self.dict_stores = {loc: values for loc, values in stores.items() if loc not in escaped}

    def _record_dict_uses(self, scope, stores):
        """Add the values @p scope stores into the dicts of @p stores; the
        locations it lets escape."""
        names = set(scope.declared)
        escaped = set()
        for node in _walk_scope(scope.body):
            if isinstance(node, _NESTED_SCOPES):
                names |= self._nested_names(node, scope)
            elif isinstance(node, ast.Name) and scope.loc(node.id) in stores:
                stored = _dict_item_use(node)
                if stored is _ESCAPES:
                    escaped.add(scope.loc(node.id))
                elif stored is not None:
                    stores[scope.loc(node.id)].append((stored, scope))
        return escaped | {scope.loc(name) for name in names} | {(MODULE, name) for name in names}

    def _compute_int_vars(self):
        """Greatest fixpoint: a variable is int when every binding of it stores
        an int, and a dict's items are ints when every value stored into it
        is, given the variables and dicts currently assumed int."""
        self.int_vars = dict.fromkeys(self.bindings, True)
        self.int_dicts = dict.fromkeys(self.dict_stores, True)
        self.bool_vars = dict.fromkeys(self.bindings, True)
        changed = True
        while changed:
            changed = False
            for facts, sources, test in ((self.int_vars, self.bindings, self.is_int),
                                         (self.int_dicts, self.dict_stores, self.is_int),
                                         (self.bool_vars, self.bindings, self.is_bool)):
                for loc, values in sources.items():
                    if facts[loc] and not all(value is not None and test(value, scope)
                                              for value, scope in values):
                        facts[loc] = False
                        changed = True

    # --- recording sites ---------------------------------------------------

    def _copy_source(self, value):
        """The list name @p value slices or copies with list() or copy(),
        else None."""
        if isinstance(value, ast.Subscript) and isinstance(value.slice, ast.Slice):
            source = value.value if self._plain_slice(value) else None
        elif isinstance(value, ast.Call) and not value.keywords:
            source = self._copied_by_call(value)
        else:
            return None
        return source if isinstance(source, ast.Name) else None

    @staticmethod
    def _plain_slice(subscript):
        return isinstance(subscript.ctx, ast.Load) and _literal_step(subscript.slice.step)

    def _copied_by_call(self, call):
        """The argument of `list(x)` or the receiver of `x.copy()`."""
        if isinstance(call.func, ast.Attribute) and call.func.attr == "copy":
            return call.func.value if not call.args else None
        if (isinstance(call.func, ast.Name) and call.func.id == "list"
                and "list" not in self.shadowed and len(call.args) == 1):
            return call.args[0]
        return None

    def _derived_list(self, value):
        """Whether @p value is `sorted(x)`, a slice of x, `list(x)` or
        `x.copy()`."""
        return bool(self._sorted_call(value)) or self._copy_source(value) is not None

    def _allocation_shape(self, value):
        """Whether @p value builds a fresh list the typed model could hold;
        its elements are checked later."""
        if isinstance(value, ast.List):
            return not any(isinstance(e, ast.Starred) for e in value.elts)
        if isinstance(value, ast.BinOp):
            return _repetition(value) is not None or (isinstance(value.op, ast.Add) and all(
                isinstance(o, ast.Name) or self._allocation_shape(o)
                for o in (value.left, value.right)))
        if self._derived_list(value):
            return True
        return (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id == _INT_BUILDER and _INT_BUILDER not in self.shadowed
                and len(value.args) == 1 and not value.keywords
                and not isinstance(value.args[0], ast.Starred))

    def _flow_into(self, loc, value, scope):
        """Record that @p value is stored to location @p loc."""
        if self._allocation_shape(value):
            self.allocations.append((scope, value, loc))
            self.classes.allocate(loc)
            # A list concatenated into a typed list is read as one, so the
            # two are typed together.
            for node in _shape_nodes(value):
                for operand in _concatenated_names(node):
                    setattr(operand, "_typed_operand", True)
                    self.classes.union(loc, scope.loc(operand.id))
        elif isinstance(value, ast.Name):
            self.classes.union(loc, scope.loc(value.id))
        elif self._analysed_call(value):
            self.classes.union(loc, (value.func.id, RETURN))
        elif not (loc[1] == RETURN and _is_none(value)):
            self.classes.poison_loc(loc, f"holds {_describe(value)}")

    def _sorted_call(self, value):
        """`sorted(x)` or `sorted(x, reverse=<bool>)` of a list name."""
        return (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id == "sorted" and "sorted" not in self.shadowed
                and len(value.args) == 1 and isinstance(value.args[0], ast.Name)
                and all(k.arg == "reverse" and isinstance(k.value, ast.Constant)
                        and isinstance(k.value.value, bool) for k in value.keywords))

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
            elif self._list_compare(node):
                self.compares.append((scope, node))
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
        func = self.top_defs[scope.name]
        self._note_annotation(ret, func.returns)
        for arg in func.args.args:
            self._note_annotation((scope.name, arg.arg), arg.annotation)

    def _note_annotation(self, loc, annotation):
        """Poison @p loc for a list annotation the typed model disagrees
        with; record the element kind `list[int]` or `list[bool]` names."""
        if annotation is None or not _is_list_annotation(annotation):
            return
        if not _allowed_list_annotation(annotation):
            self.classes.poison_loc(loc, "annotated with another list type")
        elif isinstance(annotation, ast.Subscript):
            self.annotated.append((loc, annotation.slice.id == "bool"))

    def _nested_names(self, node, scope):
        """The names nested scope @p node uses."""
        if scope.name == MODULE and node in self.top_defs.values():
            # A followed function is its own scope; only what the definition
            # evaluates here is a use.
            parts = node.decorator_list + node.args.defaults + node.args.kw_defaults
            return set().union(*(_names_in(p) for p in parts if p is not None))
        return _names_in(node)

    def _record_nested(self, node, scope):
        reason = f"used in a nested {type(node).__name__}"
        for name in self._nested_names(node, scope):
            self.classes.poison_loc(scope.loc(name), reason)
            self.classes.poison_loc((MODULE, name), reason)

    def _record_assignment(self, node, scope):
        target = _single_name_target(node)
        loc = scope.loc(target.id)
        if isinstance(node, ast.AnnAssign):
            self._note_annotation(loc, node.annotation)
        self._flow_into(loc, node.value, scope)

    def _record_return(self, node, scope):
        if scope.name == MODULE:
            return
        if node.value is not None:
            self._flow_into((scope.name, RETURN), node.value, scope)

    def _record_call(self, node, scope):
        if self._analysed_call(node):
            self.calls.append(node)
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
        if self._list_compare(parent) or _none_tested(call):
            return True
        return isinstance(parent, ast.Call) and call in parent.args and self._analysed_call(parent)

    def _list_compare(self, node):
        """`a == b` or `a != b` whose operands are each a name, a followed
        call or `sorted(x)`: compared element-wise when both are typed."""
        return (isinstance(node, ast.Compare) and len(node.ops) == 1
                and isinstance(node.ops[0], (ast.Eq, ast.NotEq)) and all(
                    isinstance(o, ast.Name) or self._analysed_call(o) or self._sorted_call(o)
                    for o in (node.left, node.comparators[0])))

    def _compared_loc(self, operand, scope):
        """The location whose typing decides operand @p operand of a list
        comparison."""
        if isinstance(operand, ast.Name):
            return scope.loc(operand.id)
        if self._analysed_call(operand):
            return (operand.func.id, RETURN)
        return scope.loc(operand.args[0].id)

    def _typed_compare(self, compare, scope):
        """Whether both operands of list comparison @p compare are typed."""
        return all(
            self.classes.typed(self._compared_loc(o, scope))
            for o in (compare.left, compare.comparators[0]))

    def _record_name(self, node, scope):
        loc = scope.loc(node.id)
        if not isinstance(node.ctx, ast.Load):
            parent = _parent(node)
            if _declaration(parent):
                self._note_annotation(loc, parent.annotation)
            elif _single_name_target(parent) is not node:
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

    def _rule_concatenated(self, node, _scope):
        """An operand of `x + y` built into a list, or the y of `x[a:b] = y`
        or `x[a:b] = reversed(y)`: it was joined to x when recorded."""
        return None if getattr(node, "_typed_operand", False) else NOT_MINE

    def _rule_compared(self, node, _scope):
        """An operand of a list comparison, checked in `_check_compares`."""
        return None if self._list_compare(_parent(node)) else NOT_MINE

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
            return self._slice_use(node, subscript, scope)
        if isinstance(subscript.ctx, ast.Store):
            value = _stored_value(subscript)
            if value is None:
                return f"element bound by {_describe(_parent(subscript))}"
        elif isinstance(subscript.ctx, ast.Load):
            value = None
        else:
            return "element deleted"
        setattr(subscript, "_typed_access", True)
        self.accesses.append((scope, subscript, value))
        return None

    def _slice_use(self, node, subscript, scope):
        """`x[a:b:c]`, a copy, or a slice store into x."""
        if isinstance(subscript.ctx, ast.Store):
            return self._slice_store(node, subscript, scope)
        if self._copy_source(subscript) is None:
            return "sliced"
        self.copies.append((scope, subscript, node))
        return None

    def _slice_store(self, node, subscript, scope):
        """`x[a:b:c] = y`, `x[a:b:c] = y[d:e:f]` or the same with
        `reversed(...)` around the value: x and y are typed together, since
        the slots are copied from one to the other. A fresh list as the value
        is an allocation into x."""
        holder = _parent(subscript)
        if not (isinstance(holder, ast.Assign) and holder.targets == [subscript]
                and _literal_step(subscript.slice.step)):
            return f"slice bound by {_describe(holder)}"
        self.slice_stores.append((scope, subscript))
        if self._allocation_shape(holder.value):
            self._flow_into(scope.loc(node.id), holder.value, scope)
            return None
        source = self._slice_store_source(holder.value)
        if source is None:
            return f"slice assigned {_describe(holder.value)}"
        if not isinstance(_parent(source), ast.Subscript):
            # A sliced y is recorded as a copy by its own use.
            setattr(source, "_typed_operand", True)
        self.classes.union(scope.loc(node.id), scope.loc(source.id))
        return None

    def _slice_store_source(self, value):
        """The list name whose elements slice store value @p value holds."""
        if self._builtin_call(value, "reversed"):
            value = value.args[0]
        if isinstance(value, ast.Subscript) and isinstance(value.slice, ast.Slice):
            value = value.value if self._plain_slice(value) else None
        return value if isinstance(value, ast.Name) else None

    def _builtin_call(self, value, name):
        """Whether @p value is `name(x)` of the builtin @p name."""
        return (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id == name and name not in self.shadowed and len(value.args) == 1
                and not value.keywords and not isinstance(value.args[0], ast.Starred))

    def _rule_none_test(self, node, _scope):
        """`x is None` and `x is not None`: a test of the pointer alone."""
        return None if _none_tested(node) else NOT_MINE

    def _rule_truth(self, node, _scope):
        """`if x`, `while x`, `not x`, `bool(x)` and the like: `len(x) != 0`."""
        parent = _parent(node)
        if isinstance(parent, _TRUTH_TESTS) and parent.test is node:
            return None
        if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
            return None
        if (isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name)
                and parent.func.id == "bool" and parent.args == [node] and not parent.keywords):
            return "passed to a shadowed bool" if "bool" in self.shadowed else None
        return NOT_MINE

    def _rule_sorted(self, node, scope):
        """`sorted(x)`: a fresh list, typed or not depending on its use."""
        call = _parent(node)
        if not (isinstance(call, ast.Call) and node in call.args
                and isinstance(call.func, ast.Name) and call.func.id == "sorted"):
            return NOT_MINE
        if not self._sorted_call(call):
            return "passed to sorted with key= or a shadowed sorted"
        parent = _parent(call)
        if (isinstance(parent, ast.Subscript) and parent.value is call
                and isinstance(parent.ctx, ast.Load) and not isinstance(parent.slice, ast.Slice)):
            setattr(parent, "_typed_access", True)
        elif not (self._result_followed(call, scope) or _compared(call)):
            return f"sorted result used in {_describe(parent)}"
        self.sorts.append((scope, call))
        return None

    def _rule_list_call(self, node, scope):
        """`list(x)`, a copy."""
        call = _parent(node)
        if not (isinstance(call, ast.Call) and node in call.args
                and self._copy_source(call) is node):
            return NOT_MINE
        self.copies.append((scope, call, node))
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

    def _method_extend(self, call, scope):
        """`x.extend(y)` of a list name or a fresh list y, which then holds
        elements of x's class."""
        if not (len(call.args) == 1 and not call.keywords and isinstance(_parent(call), ast.Expr)):
            return "method .extend used as a value"
        source = call.args[0]
        if isinstance(source, ast.Name):
            setattr(source, "_typed_operand", True)
        elif not self._allocation_shape(source):
            return f"extended by {_describe(source)}"
        self._flow_into(scope.loc(call.func.value.id), source, scope)
        self.extends.append((scope, call))
        return None

    def _method_copy(self, call, scope):
        if self._copy_source(call) is None:
            return "method .copy with arguments"
        self.copies.append((scope, call, call.func.value))
        return None

    # The uses of a list name the typed model implements, tried in order.
    USE_RULES = (_rule_concatenated, _rule_len, _rule_analysed_call, _rule_alias, _rule_list_call,
                 _rule_subscript, _rule_none_test, _rule_truth, _rule_sorted, _rule_method,
                 _rule_compared)

    # The list methods it implements.
    METHOD_RULES = {"append": _method_append, "copy": _method_copy, "extend": _method_extend}

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
            reason = self._allocation_poison(value, scope, loc)
            if reason is not None:
                poison(loc, reason)
        for scope, subscript, value in self.accesses:
            loc = scope.loc(subscript.value.id)
            if not self.is_int(subscript.slice, scope):
                poison(loc, "indexed by a non-int " + ast.unparse(subscript.slice))
            elif value is not None and not self.is_element(value, scope, loc):
                poison(loc, "stores an element of another kind")
        for scope, call in self.appends:
            loc = scope.loc(call.func.value.id)
            if not self.is_element(call.args[0], scope, loc):
                poison(loc, "appends an element of another kind")
        for loc, reason in self._untyped_bool_results():
            poison(loc, reason)
        for loc, bools in self.annotated:
            if bools != self._holds_bools(loc):
                poison(loc, "annotated with another element kind")
        changed = self._check_copies() or changed
        changed = self._check_compares() or changed
        return self._check_sorts() or changed

    def _check_compares(self):
        """Poison the typed names and call results of a list comparison
        whose other operand is not typed. A `sorted(x)` operand then becomes
        an untyped list instead. True when anything changed."""
        changed = False
        for scope, compare in self.compares:
            if self._typed_compare(compare, scope):
                continue
            for operand in (compare.left, compare.comparators[0]):
                loc = self._compared_loc(operand, scope)
                if not self._sorted_call(operand) and self.classes.typed(loc):
                    self.classes.poison_loc(loc, "compared with a list not typed")
                    changed = True
        return changed

    def _check_copies(self):
        """Poison the source of a slice, or the target of a slice store, whose
        bounds may not be ints. True when anything changed."""
        changed = False
        stores = [(scope, node, node.value) for scope, node in self.slice_stores]
        for scope, node, source in self.copies + stores:
            bounds = (node.slice.lower, node.slice.upper) if isinstance(node, ast.Subscript) else ()
            loc = scope.loc(source.id)
            if self.classes.typed(loc) and not all(b is None or self.is_int(b, scope)
                                                   for b in bounds):
                self.classes.poison_loc(loc, "sliced by a non-int")
                changed = True
        return changed

    def _check_sorts(self):
        """Poison x of a `sorted(x)` whose result the typed model cannot
        hold: read by a non-int index or stored into an untyped list. True
        when anything changed."""
        changed = False
        for scope, value, loc in self.allocations:
            source = self._sorted_call(value) and scope.loc(value.args[0].id)
            if source and self.classes.typed(source) and not self.classes.typed(loc):
                self.classes.poison_loc(source, "sorted into an untyped list")
                changed = True
        for scope, call in self.sorts:
            subscript = _parent(call)
            source = scope.loc(call.args[0].id)
            if (getattr(subscript, "_typed_access", False) and self.classes.typed(source)
                    and not self.is_int(subscript.slice, scope)):
                self.classes.poison_loc(source, "sorted and indexed by a non-int")
                changed = True
        return changed

    def _allocation_poison(self, value, scope, loc):
        """Why allocation @p value into @p loc may hold an element of another
        kind than @p loc's, else None."""
        for node in _shape_nodes(value):
            source = self._derived_source(node)
            if source is not None:
                if not self.classes.typed(scope.loc(source.id)):
                    return "copied from a list not typed"
                if self._holds_bools(scope.loc(source.id)) != self._holds_bools(loc):
                    return "copied from a list of another kind"
                continue
            repetition = _repetition(node)
            if isinstance(node,
                          ast.List) and not all(self.is_element(e, scope, loc) for e in node.elts):
                return "built from an element of another kind"
            if repetition is not None and not (self.is_element(repetition[0], scope, loc)
                                               and self.is_int(repetition[1], scope)):
                return "repeats an element of another kind or a non-int number of times"
            if isinstance(node, ast.Call) and not (self.is_int(node.args[0], scope)
                                                   and not self._holds_bools(loc)):
                return "built with a non-int size or into a bool list"
        return None

    def _derived_source(self, node):
        """The list name a copy or `sorted()` allocation reads, else None."""
        if self._sorted_call(node):
            return node.args[0]
        return self._copy_source(node)

    def _untyped_bool_results(self):
        """(location, reason) for each bool list whose copy or sorted copy
        the converter would hand to the untyped model, which reads its slots
        as ints."""
        typed_nodes = {
            id(node)
            for _, value, loc in self.allocations if self.classes.typed(loc)
            for node in _shape_nodes(value)
        }
        for _, node in self.slice_stores:
            value = _parent(node).value
            if isinstance(value, ast.Call) and isinstance(value.args[0], ast.Subscript):
                typed_nodes.add(id(value.args[0]))
        for scope, node, source in self.copies:
            if self._holds_bools(scope.loc(source.id)) and id(node) not in typed_nodes:
                yield scope.loc(source.id), "a bool list copied into an untyped list"
        for scope, call in self.sorts:
            if (self._holds_bools(scope.loc(call.args[0].id))
                    and not getattr(_parent(call), "_typed_access", False)
                    and id(call) not in typed_nodes):
                yield scope.loc(call.args[0].id), "a bool list sorted into an untyped list"

    def _seed_bool_lists(self):
        """Classes that some site fills with a bool literal, a comparison or a
        `not`, or that are annotated `list[bool]`, and the copies of them.
        Every other class holds ints; a class filled with both kinds is
        poisoned by the checks."""
        values, edges = self._stored_values()
        self.bool_roots = {self.classes.find(loc) for loc, v in values if _literal_bool(v)}
        self.bool_roots |= {self.classes.find(loc) for loc, bools in self.annotated if bools}
        changed = True
        while changed:
            grown = {self.classes.find(dst) for dst, src in edges if self._holds_bools(src)}
            changed = not grown <= self.bool_roots
            self.bool_roots |= grown

    def _stored_values(self):
        """(location, value) for each element a site stores, and (copy,
        source) for each allocation that copies or sorts a list."""
        values = [(scope.loc(s.value.id), v) for scope, s, v in self.accesses if v is not None]
        values += [(scope.loc(c.func.value.id), c.args[0]) for scope, c in self.appends]
        edges = []
        for scope, value, loc in self.allocations:
            for node in _shape_nodes(value):
                source = self._derived_source(node)
                if source is not None:
                    edges.append((loc, scope.loc(source.id)))
                elif isinstance(node, ast.List):
                    values += [(loc, e) for e in node.elts]
                elif _repetition(node) is not None:
                    values.append((loc, _repetition(node)[0]))
        return values, edges

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
        self._poison_unfollowed_bindings()
        self._collect_dict_stores()
        self._seed_bool_lists()
        while True:
            self._compute_int_vars()
            if not self._check_values():
                break
        self._flag()
        self._retype_int_reads()
        for node in ast.walk(self.module):
            node.__dict__.pop("_typed_parent", None)
            node.__dict__.pop("_typed_access", None)
            node.__dict__.pop("_typed_operand", None)
        return self

    def _flag(self):
        self._flag_compares()
        for _, value, loc in self.allocations:
            if self.classes.typed(loc):
                for node in _shape_nodes(value):
                    setattr(node, TYPED_FLAG, True)
        for scope, call in self.sorts:
            if not self.classes.typed(scope.loc(call.args[0].id)):
                continue
            subscript = _parent(call)
            if getattr(subscript, "_typed_access", False):
                setattr(subscript, TYPED_FLAG, True)
                setattr(subscript, BOOL_FLAG, self._holds_bools(scope.loc(call.args[0].id)))
                setattr(call, TYPED_FLAG, True)
            elif not getattr(call, TYPED_FLAG, False):
                setattr(call, UNTYPED_RESULT_FLAG, True)
        for scope, subscript, _ in self.accesses:
            loc = scope.loc(subscript.value.id)
            if self.classes.typed(loc):
                setattr(subscript, TYPED_FLAG, True)
                setattr(subscript, BOOL_FLAG, self._holds_bools(loc))
        for scope, call in self.appends + self.extends:
            if self.classes.typed(scope.loc(call.func.value.id)):
                setattr(call, TYPED_FLAG, True)
        # A call whose result is a typed list: a converter path that builds
        # the result itself instead of calling the function must build it
        # typed.
        for call in self.calls:
            if self.classes.typed((call.func.id, RETURN)):
                setattr(call, TYPED_FLAG, True)
        self._flag_copies()

    def _flag_compares(self):
        """Flag a comparison of two typed lists and a `sorted(x)` operand of
        one, which then builds a typed result."""
        for scope, compare in self.compares:
            if self._typed_compare(compare, scope):
                for node in (compare.left, compare.comparators[0]):
                    if self._sorted_call(node):
                        setattr(node, TYPED_FLAG, True)
                setattr(compare, TYPED_FLAG, True)

    def _retype_int_reads(self):
        """Annotate `x: Any = l[i]` of a typed l as `x: int` when every value
        x is bound to is an int, so x holds the element as an int rather than
        an untyped pointer. A loop over a list the preprocessor cannot type
        binds its target this way."""
        annotations = {}
        for scope in self.scopes:
            for node in _walk_scope(scope.body):
                target = _single_name_target(node)
                if isinstance(node, ast.AnnAssign) and target is not None:
                    annotations.setdefault(scope.loc(target.id), []).append(node)
        for loc, nodes in annotations.items():
            if not self.int_vars.get(loc, False) or not all(
                    isinstance(n.annotation, ast.Name) and n.annotation.id in ("Any", "int")
                    for n in nodes):
                continue
            for node in nodes:
                if node.annotation.id == "Any" and getattr(node.value, TYPED_FLAG, False):
                    node.annotation = ast.copy_location(ast.Name(id="int", ctx=ast.Load()),
                                                        node.annotation)

    def _flag_copies(self):
        for scope, node, source in self.copies:
            if self.classes.typed(scope.loc(source.id)):
                setattr(node, SOURCE_FLAG, True)
        for scope, node in self.slice_stores:
            if self.classes.typed(scope.loc(node.value.id)):
                setattr(node, TYPED_FLAG, True)
                # reversed(y[a:b]) copies the slice into a typed list first.
                value = _parent(node).value
                if self._builtin_call(value, "reversed") and isinstance(
                        value.args[0], ast.Subscript):
                    setattr(value.args[0], TYPED_FLAG, True)

    def builder_sites(self):
        """(typed, poison reason) for each nondet_list builder site."""
        return [(self.classes.typed(loc), self.classes.poison.get(self.classes.find(loc)))
                for _, value, loc in self.allocations if isinstance(value, ast.Call)]


def _literal_bool(value):
    """A bool literal, a comparison or a `not`."""
    return (isinstance(value, ast.Constant) and isinstance(value.value, bool)
            or isinstance(value, ast.Compare)
            or isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not))


def _repetition(node):
    """(element, count) of `[element] * count` or `count * [element]`, else
    None."""
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)):
        return None
    for seq, count in ((node.left, node.right), (node.right, node.left)):
        if isinstance(seq, ast.List) and len(
                seq.elts) == 1 and not isinstance(seq.elts[0], ast.Starred):
            return seq.elts[0], count
    return None


def _shape_nodes(value):
    """The allocations allocation @p value is built from: itself and, through
    `+`, every operand that is not a list name."""
    yield value
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
        for operand in (value.left, value.right):
            if not isinstance(operand, ast.Name):
                yield from _shape_nodes(operand)


def _concatenated_names(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return [o for o in (node.left, node.right) if isinstance(o, ast.Name)]
    return []


_ESCAPES = object()
_DICT_ANNOTATIONS = ("dict", "Dict")


def _int_dict_annotation(annotation):
    if isinstance(annotation, ast.Name):
        return annotation.id in _DICT_ANNOTATIONS
    return (isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name)
            and annotation.value.id in _DICT_ANNOTATIONS
            and isinstance(annotation.slice, ast.Tuple) and len(annotation.slice.elts) == 2
            and isinstance(annotation.slice.elts[1], ast.Name)
            and annotation.slice.elts[1].id == "int")


def _dict_item_use(name):
    """For name @p name of a dict: the value `d[k] = v` stores, None for a
    read `d[k]` or for the binding itself, _ESCAPES for any other use."""
    parent = _parent(name)
    if not isinstance(name.ctx, ast.Load):
        # The binding of a dict display; its annotation must agree.
        annotated = isinstance(parent, ast.AnnAssign)
        return _ESCAPES if annotated and not _int_dict_annotation(parent.annotation) else None
    if not (isinstance(parent, ast.Subscript) and parent.value is name
            and not isinstance(parent.slice, ast.Slice)):
        return _ESCAPES
    if isinstance(parent.ctx, ast.Load):
        return None
    holder = _parent(parent)
    if isinstance(parent.ctx, ast.Store) and isinstance(holder,
                                                        ast.Assign) and holder.targets == [parent]:
        return holder.value
    return _ESCAPES


def _compared(node):
    """Whether @p node is an operand of `==` or `!=` only."""
    parent = _parent(node)
    return isinstance(parent, ast.Compare) and all(
        isinstance(op, (ast.Eq, ast.NotEq)) for op in parent.ops)


def _list_name(node):
    """The list name of a typed access: `x` of `x[i]` or of `sorted(x)[i]`."""
    return node.args[0] if isinstance(node, ast.Call) else node


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
