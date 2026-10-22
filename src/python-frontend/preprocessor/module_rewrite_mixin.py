import ast
import copy
from collections import Counter

from preprocessor.bound_method_dispatch import dispatch_rebound_methods
from preprocessor.check_type_lowering import lower_check_type
from preprocessor.dotted_imports import alias_dotted_imports
from preprocessor.decorator_lowering import lower_user_decorators
from preprocessor.itertools_lowering import lower_itertools
from preprocessor.match_lowering import lower_match_statements
from preprocessor.namedtuple_lowering import lower_namedtuples
from preprocessor.function_value_types import reject_mixed_function_values
from preprocessor.receiver_classes import refuse_path_dependent_receivers
from preprocessor.sibling_defs import define_read_siblings_first
from preprocessor.skipped_base_init import refuse_skipped_base_init
from preprocessor.zip_lowering import inline_zip_names, lower_zip_builds
from preprocessor.unbound_locals import guard_unbound_locals
# pylint: disable=too-many-branches,too-many-boolean-expressions


def _reads_now(stmt, name):
    """Whether running `stmt` reads `name`: function and lambda bodies run
    later, so only their decorators and defaults count."""
    stack = [stmt]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load):
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stack.extend(node.decorator_list + node.args.defaults +
                         [d for d in node.args.kw_defaults if d])
        elif isinstance(node, ast.Lambda):
            stack.extend(node.args.defaults + [d for d in node.args.kw_defaults if d])
        else:
            stack.extend(ast.iter_child_nodes(node))
    return False


def _binds(stmt, name):
    """Whether the module-level statement `stmt` always rebinds `name`."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return stmt.name == name
    if isinstance(stmt, ast.Assign):
        return any(isinstance(t, ast.Name) and t.id == name for t in stmt.targets)
    if isinstance(stmt, ast.AnnAssign):
        return stmt.value is not None and isinstance(stmt.target,
                                                     ast.Name) and stmt.target.id == name
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return any((a.asname or a.name.split(".")[0]) == name for a in stmt.names)
    return False


def _rename_shadowed_classes(module):
    """Rename a module-level class that a later module-level statement rebinds
    before anything reads it, as in `class C: ...` then `C = f`. Every read of
    `C` then means the new binding, which the converter would otherwise take
    for the class (#8343)."""
    taken = {n.id for n in ast.walk(module) if isinstance(n, ast.Name)}
    for i, stmt in enumerate(module.body):
        if not isinstance(stmt, ast.ClassDef):
            continue
        for later in module.body[i + 1:]:
            if _reads_now(later, stmt.name):
                break
            if _binds(later, stmt.name):
                renamed = f"{stmt.name}__esbmc_shadowed"
                while renamed in taken:
                    renamed += "_"
                taken.add(renamed)
                stmt.name = renamed
                break


def _referenced_names(module):
    """The names `module` reads, deletes or updates in place."""
    return {
        n.id
        for n in ast.walk(module) if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Store)
    } | {
        n.target.id
        for n in ast.walk(module)
        if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name)
    }


class ModuleRewriteMixin:
    called_names: set
    shadowed_nondet_collections: set
    imported_module_names: set
    variable_annotations: dict
    module_dunder_all: list | None
    exported_range_aliases: set
    exported_range_wrappers: dict

    @staticmethod
    def _target_names(target):
        """Names bound by an assignment target, including tuple/list unpacking."""
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, ast.Starred):
            return ModuleRewriteMixin._target_names(target.value)
        if isinstance(target, (ast.Tuple, ast.List)):
            return {n for e in target.elts for n in ModuleRewriteMixin._target_names(e)}
        return set()

    @staticmethod
    def _rebound_module_names(module_node):  # pylint: disable=too-many-branches
        """Module-top names that may be rebound after their first binding."""
        seen = set()
        rebound = set()
        target_names = ModuleRewriteMixin._target_names

        def _collect_bindings(stmt):
            if isinstance(stmt, ast.Assign):
                names = set()
                for t in stmt.targets:
                    names |= target_names(t)
                return names
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                return {stmt.target.id}
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return {stmt.name}
            if isinstance(stmt, ast.Import):
                return {a.asname or a.name.split(".")[0] for a in stmt.names}
            if isinstance(stmt, ast.ImportFrom):
                return {a.asname or a.name for a in stmt.names}
            return set()

        for stmt in module_node.body:
            if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                rebound.add(stmt.target.id)
                seen.add(stmt.target.id)
                continue
            for name in _collect_bindings(stmt):
                if name in seen:
                    rebound.add(name)
                seen.add(name)

        def _walk_module_only(root):
            for child in ast.iter_child_nodes(root):
                if isinstance(child, ast.Lambda):
                    continue
                yield child
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    yield from _walk_module_only(child)

        flow_nodes = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)
        for stmt in module_node.body:
            if isinstance(stmt, flow_nodes):
                for inner in _walk_module_only(stmt):
                    rebound |= ModuleRewriteMixin._bound_names_of(inner)

        for inner in ast.walk(module_node):
            if isinstance(inner, ast.Global):
                rebound.update(inner.names)

        return rebound

    @staticmethod
    def _bound_names_of(node):
        """Names a single statement/expression node binds in its own scope."""
        target_names = ModuleRewriteMixin._target_names
        if isinstance(node, ast.Assign):
            return set().union(*(target_names(t) for t in node.targets))
        if isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            return {node.target.id} if isinstance(node.target, ast.Name) else set()
        if isinstance(node, (ast.For, ast.AsyncFor)):
            return target_names(node.target)
        if isinstance(node, (ast.With, ast.AsyncWith)):
            return set().union(*(target_names(i.optional_vars) for i in node.items
                                 if i.optional_vars))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return {node.name}
        return set()

    @staticmethod
    def _scope_locally_binds(scope_node, names):  # pylint: disable=too-many-branches
        """True iff any of *names* is locally bound inside *scope_node*."""
        if not isinstance(scope_node,
                          (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            return False

        target_names = ModuleRewriteMixin._target_names

        if isinstance(scope_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = scope_node.args
            for p in (*a.args, *a.posonlyargs, *a.kwonlyargs):
                if p.arg in names:
                    return True
            if a.vararg and a.vararg.arg in names:
                return True
            if a.kwarg and a.kwarg.arg in names:
                return True

        body = scope_node.body
        body = body if isinstance(body, list) else [body]
        stack = list(body)
        while stack:
            n = stack.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if n.name in names:
                    return True
                continue
            if isinstance(n, ast.Lambda):
                continue
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if target_names(t) & names:
                        return True
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)):
                if isinstance(n.target, ast.Name) and n.target.id in names:
                    return True
            elif isinstance(n, ast.NamedExpr):
                if isinstance(n.target, ast.Name) and n.target.id in names:
                    return True
            elif isinstance(n, (ast.For, ast.AsyncFor)):
                if target_names(n.target) & names:
                    return True
            elif isinstance(n, (ast.With, ast.AsyncWith)):
                for item in n.items:
                    if item.optional_vars and target_names(item.optional_vars) & names:
                        return True
            elif isinstance(n, ast.Import):
                for al in n.names:
                    if (al.asname or al.name.split(".")[0]) in names:
                        return True
            elif isinstance(n, ast.ImportFrom):
                for al in n.names:
                    if (al.asname or al.name) in names:
                        return True
            elif isinstance(n, (ast.Global, ast.Nonlocal)):
                if any(name in names for name in n.names):
                    return True
            for child in ast.iter_child_nodes(n):
                stack.append(child)
        return False

    @classmethod
    def _iter_module_scope(cls, root, shadow_names):
        """Yield descendants of *root* whose enclosing scope is the module."""
        for child in ast.iter_child_nodes(root):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                if cls._scope_locally_binds(child, shadow_names):
                    continue
            yield child
            yield from cls._iter_module_scope(child, shadow_names)

    @staticmethod
    def collect_range_aliases(module_node, seed=frozenset()):
        """Collect module-scope canonical-range aliases."""
        rebound = ModuleRewriteMixin._rebound_module_names(module_node)
        all_aliases = set(seed) - rebound
        local = set()
        changed = True
        while changed:
            changed = False
            for stmt in module_node.body:
                if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(
                        stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Name)):
                    continue
                lhs = stmt.targets[0].id
                rhs = stmt.value.id
                if lhs in all_aliases or lhs in rebound:
                    continue
                if rhs == "range" or rhs in all_aliases:
                    all_aliases.add(lhs)
                    local.add(lhs)
                    changed = True
        return local, all_aliases

    def apply_range_rewrites(self, module_node, alias_seed=frozenset(), wrapper_seed=None):
        """Run the range-alias and range-wrapper rewrites on *module_node*."""
        self._rewrite_range_aliases(module_node, seed=alias_seed)
        self._inline_range_wrappers(module_node, seed=wrapper_seed)
        self._fold_module_const_range_args(module_node)

    @staticmethod
    def _int_literal_value(node):
        """Int value of an integer-literal AST node (or unary +/- of one), else
        None. Booleans are excluded — ``True``/``False`` are not range sizes."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int) \
                and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant) \
                and isinstance(node.operand.value, int) \
                and not isinstance(node.operand.value, bool):
            if isinstance(node.op, ast.USub):
                return -node.operand.value
            if isinstance(node.op, ast.UAdd):
                return node.operand.value
        return None

    def _collect_module_const_ints(self, module_node):
        """Names bound exactly once, unconditionally at module top level, to an
        integer literal, and never rebound — genuine module constants.

        ``_rebound_module_names`` already flags names assigned more than once,
        assigned inside any conditional/loop/``with``, or declared ``global``,
        so excluding it leaves only names that hold one fixed integer on every
        path. Comprehension / generator-expression loop targets are a separate
        local scope that ``_rebound_module_names`` does not see, so they are
        excluded explicitly: a same-named module constant must never be folded
        into ``[... range(N) ... for N in ...]``, where ``N`` is the loop
        variable, not the constant."""
        rebound = self._rebound_module_names(module_node)
        # Names bound in a scope/position _rebound_module_names does not track:
        # comprehension/genexp loop targets (their own local scope) and walrus
        # (``:=``) targets. A same-named module constant must not be folded into
        # any of these — the name may hold a different value there.
        local_binds = set()
        for node in ast.walk(module_node):
            if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                for gen in node.generators:
                    local_binds |= self._target_names(gen.target)
            elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
                local_binds.add(node.target.id)
        consts = {}
        for stmt in module_node.body:
            if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)):
                name = stmt.targets[0].id
                value = self._int_literal_value(stmt.value)
                if (value is not None and name not in rebound and name not in local_binds):
                    consts[name] = value
        return consts

    def _fold_module_const_range_args(self, module_node):
        """Replace module-constant ``Name`` args of ``range()`` calls with their
        integer literals.

        ``list(range(N))`` with a constant ``N`` otherwise takes the *symbolic*
        range path, which builds a list whose ``size`` is set but whose elements
        are nondet garbage — reading/slicing/copying it raises a spurious
        ``dereference failure`` (an unsound false positive). Folding a genuine
        constant so the *concrete* range path materialises real elements removes
        the false alarm. ``_iter_module_scope`` skips any nested function/class
        scope that locally rebinds the name, so a shadowing parameter/local is
        never folded to the module constant.

        Only ``list(range(...))`` / ``tuple(range(...))`` — the sequence
        *materialisation* contexts where the false deref occurs — are folded. A
        ``range()`` used directly as a ``for`` iterable is left untouched: it
        never materialises elements, and its lowering models a zero ``step`` as
        a runtime ``step != 0`` assertion that folding to a literal would turn
        into a preprocess-time error (regression/python/range23_fail)."""
        consts = self._collect_module_const_ints(module_node)
        if not consts:
            return
        for n in self._iter_module_scope(module_node, set(consts)):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id in ("list", "tuple") and len(n.args) == 1 and not n.keywords):
                continue
            inner = n.args[0]
            if not (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
                    and inner.func.id == "range" and not inner.keywords):
                continue
            for i, arg in enumerate(inner.args):
                if isinstance(arg, ast.Name) and arg.id in consts:
                    folded = ast.Constant(value=consts[arg.id])
                    ast.copy_location(folded, arg)
                    inner.args[i] = folded

    def _rewrite_range_aliases(self, module_node, seed=frozenset()):
        """Rewrite module-scope alias = range (and chains) to canonical range."""
        local, all_aliases = self.collect_range_aliases(module_node, seed)
        self.exported_range_aliases |= local
        self.exported_range_aliases |= (set(seed) & all_aliases)

        if not all_aliases:
            return

        for n in self._iter_module_scope(module_node, all_aliases):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id in all_aliases):
                n.func.id = "range"

        if not local:
            return

        module_node.body = [
            stmt for stmt in module_node.body
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id in local
                    and isinstance(stmt.value, ast.Name) and
                    (stmt.value.id == "range" or stmt.value.id in all_aliases))
        ]

    @staticmethod
    def _simple_single_return_defs(module_node):
        """Yield ``(stmt, params, expr)`` for each undecorated, never-rebound
        module-level ``def f(p, ...): return <expr>`` with plain parameters."""
        rebound = ModuleRewriteMixin._rebound_module_names(module_node)
        for stmt in module_node.body:
            if not (isinstance(stmt, ast.FunctionDef) and stmt.name not in rebound
                    and not stmt.decorator_list and len(stmt.body) == 1
                    and isinstance(stmt.body[0], ast.Return) and stmt.body[0].value is not None):
                continue
            args = stmt.args
            if (args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs or args.defaults
                    or args.kw_defaults):
                continue
            yield stmt, [a.arg for a in args.args], stmt.body[0].value

    @staticmethod
    def _collect_single_return_funcs(module_node):
        """One-parameter single-return defs as name -> (param, expr), so a
        ``key=f`` folds like a lambda."""
        return {
            stmt.name: (params[0], expr)
            for stmt, params, expr in ModuleRewriteMixin._simple_single_return_defs(module_node)
            if len(params) == 1
        }

    def _collect_param_subscripting_funcs(self, module_node):
        """Names that may index the first argument they are called with.

        Covers defs whose body subscripts their first parameter, plus every
        imported name, whose body this scan cannot see. Keyed on name alone, so
        a nested or rebound def of the same name counts too: over-refusing
        costs a clear error, under-refusing costs a verdict.
        """
        names = set()
        for stmt in ast.walk(module_node):
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                names.update(alias.asname or alias.name for alias in stmt.names)
            elif (isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.args.args
                  and self.subscripts_name(stmt, stmt.args.args[0].arg)):
                names.add(stmt.name)
        return names

    @staticmethod
    def collect_range_wrappers(module_node):
        """Collect trivial def f(p): return range(...) wrappers at module scope."""
        wrappers = {}
        for stmt, params, call in ModuleRewriteMixin._simple_single_return_defs(module_node):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id == "range" and not call.keywords):
                continue
            param_set = set(params)
            if not all((isinstance(a, ast.Name) and a.id in param_set) or (isinstance(
                    a, ast.Constant) and isinstance(a.value, int) and not isinstance(a.value, bool))
                       for a in call.args):
                continue
            wrappers[stmt.name] = (params, call.args)
        return wrappers

    def _inline_range_wrappers(self, module_node, seed=None):
        """Inline trivial def f(p): return range(p_or_const, ...) wrappers."""
        local = self.collect_range_wrappers(module_node)
        self.exported_range_wrappers.update(local)

        wrappers = dict(seed) if seed else {}
        rebound = self._rebound_module_names(module_node)
        for name in list(wrappers):
            if name in rebound and name not in local:
                del wrappers[name]
        self.exported_range_wrappers.update(wrappers)
        wrappers.update(local)

        if not wrappers:
            return

        wrapper_names = set(wrappers)
        for n in self._iter_module_scope(module_node, wrapper_names):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id in wrappers):
                continue
            if n.keywords or any(isinstance(a, ast.Starred) for a in n.args):
                continue
            params, template = wrappers[n.func.id]
            if len(n.args) != len(params):
                continue
            subst = dict(zip(params, n.args))
            n.args = [
                copy.deepcopy(subst[t.id]) if isinstance(t, ast.Name) else copy.deepcopy(t)
                for t in template
            ]
            n.func.id = "range"
            ast.fix_missing_locations(n)

    @staticmethod
    def _capture_dunder_all(module_node):
        """Return the literal list/tuple of names assigned to __all__."""
        for stmt in module_node.body:
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == "__all__"):
                continue
            value = stmt.value
            if not isinstance(value, (ast.List, ast.Tuple)):
                return None
            names = []
            for el in value.elts:
                if isinstance(el, ast.Constant) and isinstance(el.value, str):
                    names.append(el.value)
                else:
                    return None
            return names
        return None

    @staticmethod
    def _binding_sites(module_node):
        """How many places anywhere in *module_node* bind or declare each name."""
        sites = Counter()
        for n in ast.walk(module_node):
            if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load):
                sites[n.id] += 1
            elif isinstance(n, ast.alias):
                sites[n.asname or n.name.split(".")[0]] += 1
            elif isinstance(n, ast.arg):
                sites[n.arg] += 1
            elif isinstance(n, (ast.Global, ast.Nonlocal)):
                sites.update(n.names)
            elif isinstance(getattr(n, "name", None), str):
                sites[n.name] += 1
            elif isinstance(n, ast.MatchMapping) and n.rest:
                sites[n.rest] += 1
        return sites

    @staticmethod
    def _appears_before(module_node, name, stmt):
        """Whether a Name *name* occurs in *module_node* before *stmt*."""
        positions = ((getattr(n, "lineno", 0), getattr(n, "col_offset", 0))
                     for n in ast.walk(module_node) if isinstance(n, ast.Name) and n.id == name)
        return any(p < (stmt.lineno, stmt.col_offset) for p in positions)

    def _rewrite_module_aliases(self, module_node):
        """Rewrite a top-level `alias = mod` into `import <module> as alias`.

        Only when `mod` is bound solely by an earlier top-level import, `alias`
        solely by this statement, and nothing reads `alias` before it: then
        both forms bind the same module at the same point, and the converter
        resolves calls through the import form only (#8366)."""
        sites = self._binding_sites(module_node)
        modules = {}
        for i, stmt in enumerate(module_node.body):
            if isinstance(stmt, ast.Import):
                for a in stmt.names:
                    root = a.name.split(".")[0]
                    modules[a.asname or root] = a.name if a.asname else root
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Name)):
                continue
            alias, name = stmt.targets[0].id, stmt.value.id
            if (name not in modules or sites[name] != 1 or sites[alias] != 1
                    or self._appears_before(module_node, alias, stmt)):
                continue
            imp = ast.Import(names=[ast.alias(name=modules[name], asname=alias)])
            module_node.body[i] = ast.fix_missing_locations(ast.copy_location(imp, stmt))

    def prepare_module(self, node, alias_seed=frozenset(), wrapper_seed=None):
        """Run pre-visit analyses and range-alias/wrapper canonicalization."""
        lower_check_type(node, self.module_name)
        _rename_shadowed_classes(node)
        alias_dotted_imports(node)
        inline_zip_names(node)
        lower_zip_builds(node)
        reject_mixed_function_values(node, self.module_name)
        lower_match_statements(node)
        lower_itertools(node)
        lower_namedtuples(node, self.module_name)
        lower_user_decorators(node, self.module_name)
        define_read_siblings_first(node)
        refuse_path_dependent_receivers(node, self.module_name)
        refuse_skipped_base_init(node, self.module_name)
        self.expand_unittest_main(node)
        self._rewrite_module_aliases(node)
        self._rewrite_map_values(node)
        guard_unbound_locals(node)
        self.in_order_bindings.update(
            dispatch_rebound_methods(node, self._names_bound_by, self._variable_names(node),
                                     self._next_unpack_tmp_id, self.module_name))
        self.hoist_nested_classes(node)
        self.copy_aliased_methods(node)

        for n in ast.walk(node):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                self.called_names.add(n.func.id)
        self.referenced_names = _referenced_names(node)
        self._scan_fixed_receivers(node)
        self.variable_names = self._variable_names(node)
        self.scope_binding_counts = self._scope_binding_counts(node)

        self._scan_shadowed_nondet_collections(node)
        self._scan_imported_module_names(node)
        self._scan_functools_reduce_names(node)

        self.iterator_classes = self._scan_iterator_classes(node)
        self.iterator_params = self._scan_iterator_params(node)

        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        annotation_node = self._create_annotation_node_from_value(stmt.value)
                        if annotation_node:
                            self.variable_annotations[target.id] = annotation_node
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                self.variable_annotations[stmt.target.id] = stmt.annotation

        self._scan_dict_literal_bindings_and_calls(node)
        self.module_class_names, self.instance_var_classes = \
            self._build_var_class_map(node)
        self.attr_list_element_classes = self._scan_attr_list_element_classes(node)
        self.list_var_element_classes = self._scan_list_var_element_classes(node)
        self.module_dunder_all = self._capture_dunder_all(node)
        self._scan_sequence_iterators(node)
        self._scan_starred_calls(node)
        self._scan_generator_consumers(node)
        self._scan_module_vararg_defs(node)
        self._single_return_funcs = self._collect_single_return_funcs(node)
        self._param_subscripting_funcs = self._collect_param_subscripting_funcs(node)
        self.apply_range_rewrites(node, alias_seed=alias_seed, wrapper_seed=wrapper_seed)

    def _scan_iterator_classes(self, node):
        """Classes whose instances a for loop iterates through `__iter__`, for visit_For.

        A class qualifies when it defines `__iter__` or inherits it from a
        module class that does. A class without `__iter__` is not iterable,
        and CPython's TypeError for it is already reported. The class name
        must be bound only by its class statement, so a call of it constructs
        the class."""
        bind_counts = {}
        for n in ast.walk(node):
            for name in self._bound_names(n):
                bind_counts[name] = bind_counts.get(name, 0) + 1
        classes = [n for n in ast.walk(node) if isinstance(n, ast.ClassDef)]
        iterable = set()
        grew = True
        while grew:
            grew = False
            for c in classes:
                if c.name not in iterable and self._has_iter(c, iterable):
                    iterable.add(c.name)
                    grew = True
        return {name for name in iterable if bind_counts.get(name) == 1}

    def _scan_iterator_params(self, node):
        """`(id(def), param)` for each parameter of a module function that
        every reference to the function passes a `Cls(...)` of an iterator
        class, so a for loop over the parameter can use `__iter__` (#8628).

        The function name must be bound once and only ever called directly,
        so no call goes unseen."""
        counts = Counter(name for n in ast.walk(node) for name in self._names_bound_by(n))
        loads = Counter(n.id for n in ast.walk(node)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
        calls = {}
        for n in ast.walk(node):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                calls.setdefault(n.func.id, []).append(n)
        return {(id(func), param)
                for func in node.body if isinstance(func, ast.FunctionDef)
                and counts[func.name] == 1 and loads[func.name] == len(calls.get(func.name, ())) > 0
                for param in self._iterator_params_of(func, calls[func.name])}

    def _iterator_params_of(self, func, calls):
        """The plain positional parameters of `func` that each of `calls`
        passes positionally as a `Cls(...)` of an iterator class."""
        if func.args.vararg or func.args.kwarg or func.decorator_list or any(
                c.keywords or any(isinstance(a, ast.Starred) for a in c.args) for c in calls):
            return []
        params = func.args.posonlyargs + func.args.args
        return [
            p.arg for i, p in enumerate(params)
            if all(i < len(c.args) and self._is_iterator_construction(c.args[i]) for c in calls)
        ]

    def _is_iterator_construction(self, expr):
        return (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)
                and expr.func.id in self.iterator_classes)

    @staticmethod
    def _has_iter(cls, iterable):
        return any(isinstance(m, ast.FunctionDef) and m.name == "__iter__"
                   for m in cls.body) or any(
                       isinstance(b, ast.Name) and b.id in iterable for b in cls.bases)

    def _scan_dict_literal_bindings_and_calls(self, node):
        """Collect evidence for parameter-dict element recovery (#5444).

        For each scope (the module body and every function body, independently)
        records its dict-literal name bindings, then resolves each direct
        ``name(...)`` call's positional arguments against that scope's bindings
        (falling back to the module scope for free names). The result maps a
        function name to its per-call list of argument dict[K, V] shapes, which
        _recover_param_dict_annotation collapses only when every call agrees —
        keeping the inference sound and scope-correct (a function-local dict
        never poisons a same-named module global).
        """
        self._dict_param_call_shapes = {}
        module_binds, module_locals = self._collect_scope_dict_binds(node.body)
        self._record_scope_calls(node.body, module_binds, module_locals, module_binds)
        for n in ast.walk(node):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                local_binds, local_names = self._collect_scope_dict_binds(n.body)
                self._record_scope_calls(n.body, local_binds, local_names, module_binds)

    def _scan_imported_module_names(self, node):
        """Record names bound by a module-level `import` and by nothing else.

        Only such a name denotes the module wherever it is read, so only its
        attributes may stand for the module's nondet functions.
        """
        imports, others = [], set()
        for n in ast.walk(node):
            if isinstance(n, ast.Import):
                bound = [a.asname or a.name.split(".")[0] for a in n.names]
                if n in node.body:
                    imports += bound
                else:
                    others.update(bound)
            else:
                others |= self._names_bound_without_import(n)
        self.imported_module_names = {m for m in imports if imports.count(m) == 1} - others

    @staticmethod
    def _names_bound_without_import(n):
        """Names `n` binds in its scope, for any node but an `import`."""
        if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load):
            return {n.id}
        if isinstance(n, ast.arg):
            return {n.arg}
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return {n.name}
        if isinstance(n, ast.ImportFrom):
            return {a.asname or a.name for a in n.names}
        if isinstance(n, ast.ExceptHandler) and n.name:
            return {n.name}
        return set()

    def _scan_shadowed_nondet_collections(self, node):
        """Record `nondet_list`/`nondet_dict` names this module defines itself.

        The rewrite to a typed builder matches on the callee name, so without
        this a user's own `def nondet_list(...)` would be replaced by ESBMC's
        model and its body never verified. A name that is only *imported* stays
        interceptable: SV-COMP harnesses import `_sv_verifier.nondet_list`
        expecting ESBMC's model, not the stub's body.
        """
        intrinsics = ("nondet_list", "nondet_dict")

        def parameter_names(fn_args):
            """Every name a parameter list binds, across all five kinds."""
            if fn_args is None:
                return
            for arg in (list(getattr(fn_args, "posonlyargs", [])) + list(fn_args.args) +
                        list(fn_args.kwonlyargs)):
                yield arg.arg
            for arg in (fn_args.vararg, fn_args.kwarg):
                if arg is not None:
                    yield arg.arg

        for n in ast.walk(node):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                if getattr(n, "name", None) in intrinsics:
                    self.shadowed_nondet_collections.add(n.name)
                for name in parameter_names(getattr(n, "args", None)):
                    if name in intrinsics:
                        self.shadowed_nondet_collections.add(name)
            elif isinstance(n, ast.Assign):
                for target in n.targets:
                    if isinstance(target, ast.Name) and target.id in intrinsics:
                        self.shadowed_nondet_collections.add(target.id)
            elif isinstance(n, ast.AnnAssign):
                if isinstance(n.target, ast.Name) and n.target.id in intrinsics:
                    self.shadowed_nondet_collections.add(n.target.id)

    @staticmethod
    def _walk_scope(stmts):
        """Yield AST nodes within ``stmts`` without crossing into a nested
        function/class/lambda scope (which binds its own names)."""
        stack = list(stmts)
        while stack:
            n = stack.pop()
            yield n
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            stack.extend(ast.iter_child_nodes(n))

    def _collect_scope_dict_binds(self, stmts):
        """Map names bound to a dict literal in this scope to their dict[K, V]
        annotation. A name reassigned to a conflicting shape or to a non-dict is
        poisoned (dropped) so it is never used as recovery evidence. Returns
        (bindings, all-assigned-names)."""
        binds = {}
        poisoned = set()
        assigned = set()
        for n in self._walk_scope(stmts):
            if isinstance(n, ast.Assign):
                names = [t.id for t in n.targets if isinstance(t, ast.Name)]
                value = n.value
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                names = [n.target.id]
                value = n.value
            else:
                continue
            ann = (self._build_dict_annotation_from_literal(value)
                   if isinstance(value, ast.Dict) else None)
            for name in names:
                assigned.add(name)
                if name in poisoned:
                    continue
                if ann is None or (name in binds and ast.dump(binds[name]) != ast.dump(ann)):
                    poisoned.add(name)
                    binds.pop(name, None)
                else:
                    binds[name] = ann
        return binds, assigned

    def _record_scope_calls(self, stmts, local_binds, local_names, module_binds):
        """Record the per-call positional dict shapes for every direct call in
        this scope, resolving Name arguments against the local bindings (or the
        module bindings for free names)."""
        for n in self._walk_scope(stmts):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)):
                continue
            # A *splat shifts every later argument's runtime position by
            # len(iterable), unknowable statically -- recording syntactic
            # indices would let _recover_param_dict_annotation size the
            # WRONG parameter. Refuse the whole call (None at every index).
            if any(isinstance(a, ast.Starred) for a in n.args):
                self._dict_param_call_shapes.setdefault(n.func.id, []).append([None] * len(n.args))
                continue
            shapes = []
            for arg in n.args:
                if isinstance(arg, ast.Dict):
                    shapes.append(self._build_dict_annotation_from_literal(arg))
                elif isinstance(arg, ast.List):
                    # list-of-tuple-literals: shape evidence for sizing the
                    # str components of a list[tuple[...]] parameter
                    # annotation (#5571), same table, same all-sites-agree
                    # consumers.
                    shapes.append(self._build_list_annotation_from_literal(arg))
                elif isinstance(arg, ast.Name):
                    if arg.id in local_names:
                        shapes.append(local_binds.get(arg.id))
                    else:
                        shapes.append(module_binds.get(arg.id))
                else:
                    shapes.append(None)
            self._dict_param_call_shapes.setdefault(n.func.id, []).append(shapes)
