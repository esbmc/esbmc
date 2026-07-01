import ast
import copy
# pylint: disable=too-many-branches,too-many-boolean-expressions


class ModuleRewriteMixin:
    called_names: set
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
                if isinstance(child,
                              (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                    continue
                yield child
                yield from _walk_module_only(child)

        for stmt in module_node.body:
            flow_nodes = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With,
                          ast.AsyncWith)
            if not isinstance(stmt, flow_nodes):
                continue
            for inner in _walk_module_only(stmt):
                if isinstance(inner, ast.Assign):
                    for t in inner.targets:
                        rebound |= target_names(t)
                elif isinstance(inner, (ast.AnnAssign, ast.AugAssign)):
                    if isinstance(inner.target, ast.Name):
                        rebound.add(inner.target.id)
                elif isinstance(inner, ast.NamedExpr) and isinstance(inner.target, ast.Name):
                    rebound.add(inner.target.id)
                elif isinstance(inner, (ast.For, ast.AsyncFor)):
                    rebound |= target_names(inner.target)
                elif isinstance(inner, (ast.With, ast.AsyncWith)):
                    for item in inner.items:
                        if item.optional_vars:
                            rebound |= target_names(item.optional_vars)

        for inner in ast.walk(module_node):
            if isinstance(inner, ast.Global):
                rebound.update(inner.names)

        return rebound

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
    def collect_range_wrappers(module_node):  # pylint: disable=too-many-boolean-expressions
        """Collect trivial def f(p): return range(...) wrappers at module scope."""
        rebound = ModuleRewriteMixin._rebound_module_names(module_node)
        wrappers = {}
        for stmt in module_node.body:
            if not (isinstance(stmt, ast.FunctionDef) and stmt.name not in rebound
                    and len(stmt.body) == 1 and isinstance(stmt.body[0], ast.Return)):
                continue
            call = stmt.body[0].value
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id == "range" and not call.keywords):
                continue
            args = stmt.args
            has_non_simple_args = (args.vararg is not None or args.kwarg is not None
                                   or args.kwonlyargs or args.posonlyargs or args.defaults
                                   or args.kw_defaults)
            if has_non_simple_args:
                continue
            params = [a.arg for a in args.args]
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

    def prepare_module(self, node, alias_seed=frozenset(), wrapper_seed=None):
        """Run pre-visit analyses and range-alias/wrapper canonicalization."""
        for n in ast.walk(node):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                self.called_names.add(n.func.id)

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
        self.apply_range_rewrites(node, alias_seed=alias_seed, wrapper_seed=wrapper_seed)

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
            shapes = []
            for arg in n.args:
                if isinstance(arg, ast.Dict):
                    shapes.append(self._build_dict_annotation_from_literal(arg))
                elif isinstance(arg, ast.Name):
                    if arg.id in local_names:
                        shapes.append(local_binds.get(arg.id))
                    else:
                        shapes.append(module_binds.get(arg.id))
                else:
                    shapes.append(None)
            self._dict_param_call_shapes.setdefault(n.func.id, []).append(shapes)
