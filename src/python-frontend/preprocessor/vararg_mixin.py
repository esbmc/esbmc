"""Lowering of functions declaring *args into fixed-arity specializations."""

import ast
import copy


class VarargMixin:

    def _scan_module_vararg_defs(self, node):
        """Mark the variadic defs that live directly in the module body."""
        self._vararg_module_defs = {
            id(stmt)
            for stmt in node.body if isinstance(stmt, ast.FunctionDef) and stmt.args.vararg
        }

    def _enter_vararg_scope(self, node):
        """Open the scope of `node`, returning the def table to restore on exit.

        A variadic def nested in `node` is visible only while that scope is
        open, so a sibling function cannot resolve a call against it.
        """
        self._vararg_scope_stack.append(node)
        return dict(self._vararg_func_defs)

    def _exit_vararg_scope(self, node, saved_func_defs):
        # Keyed by the def that was pushed: a visitor may hand back a rewritten
        # node, and ownership was recorded against the original.
        scope = self._vararg_scope_stack.pop()
        own_entry = self._vararg_func_defs.get(scope.name)
        self._vararg_func_defs = saved_func_defs
        # Only the defs nested inside `scope` go out of scope here; `scope`
        # itself binds its own name in the scope that encloses it.
        if own_entry is scope:
            self._vararg_func_defs[scope.name] = scope
        self._inject_vararg_specializations(node, id(scope))

    def _enclosing_vararg_scope(self):
        """The def enclosing the one being visited, which is itself on top."""
        return self._vararg_scope_stack[-2] if len(self._vararg_scope_stack) > 1 else None

    def _record_vararg_function(self, node, qualified_name):
        if node.args.vararg:
            self.functionVarargs.add(qualified_name)
            # Registered only once visited, since a specialization is a copy of
            # the body as the converter will see it -- an unvisited body still
            # holds constructs (`for`, comprehensions) with no lowering there.
            # A specialization takes the place of the def it copies, so only a
            # def that is a statement of its own scope's body has somewhere to go.
            owner = self._enclosing_vararg_scope()
            if (any(stmt is node for stmt in owner.body)
                    if owner is not None else id(node) in self._vararg_module_defs):
                self._vararg_func_defs[node.name] = node
                self._vararg_def_owners[id(node)] = owner
        else:
            self.functionVarargs.discard(qualified_name)
            self._vararg_func_defs.pop(qualified_name, None)

    @staticmethod
    def _name_is_used(body, name):
        return any(
            isinstance(inner, ast.Name) and inner.id == name for stmt in body
            for inner in ast.walk(stmt))

    def _is_specializable_vararg_call(self, node, func_def):
        if node.keywords or func_def.decorator_list or func_def.args.kwonlyargs:
            return False
        if func_def.name in self.generator_func_defs:
            return False
        if any(isinstance(arg, ast.Starred) for arg in node.args):
            return False
        if len(node.args) < len(func_def.args.args):
            return False
        if func_def.args.kwarg and self._name_is_used(func_def.body, func_def.args.kwarg.arg):
            return False
        # Copying a nested definition once per arity would emit clashing symbols.
        if any(
                isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                for stmt in func_def.body for inner in ast.walk(stmt)):
            return False
        # A recursive call inside the copied body would still target the
        # variadic original, so the copy would not be self-contained.
        return not self._name_is_used(func_def.body, func_def.name)

    def _specialize_vararg_call(self, node):
        if not isinstance(node.func, ast.Name):
            return
        func_def = self._vararg_func_defs.get(node.func.id)
        if func_def is None or not self._name_is_used(func_def.body, func_def.args.vararg.arg):
            return
        if not self._is_specializable_vararg_call(node, func_def):
            return

        owner = self._vararg_def_owners.get(id(func_def))
        extra = len(node.args) - len(func_def.args.args)
        # The owner prefix keeps same-named defs nested in different functions
        # from sharing one specialization, which lives in only one of them.
        prefix = f"{owner.name}__" if owner is not None else ""
        spec_name = f"{prefix}{func_def.name}__esbmc_va{extra}"
        if spec_name not in self._vararg_specializations:
            self._vararg_specializations[spec_name] = self._build_vararg_specialization(
                func_def, spec_name, extra)
            owner_key = id(owner) if owner is not None else None
            owned = self._vararg_owner_specs.setdefault(owner_key, {})
            owned.setdefault(id(func_def), []).append(spec_name)
        node.func = ast.Name(id=spec_name, ctx=ast.Load())
        self._vararg_dropped_defs.add(id(func_def))
        ast.fix_missing_locations(node)

    def _build_vararg_specialization(self, func_def, spec_name, extra):
        spec = copy.deepcopy(func_def)
        spec.name = spec_name
        pad = [f"__esbmc_va_{i}" for i in range(extra)]
        spec.args.args = spec.args.args + [ast.arg(arg=name) for name in pad]
        vararg_name = spec.args.vararg.arg
        spec.args.vararg = None
        spec.args.kwarg = None
        elts = [ast.Name(id=name, ctx=ast.Load()) for name in pad]
        packed = ast.Tuple(elts=elts, ctx=ast.Load())
        pack = ast.Assign(targets=[ast.Name(id=vararg_name, ctx=ast.Store())], value=packed)
        spec.body = [pack] + spec.body
        self.functionParams[spec_name] = [arg.arg for arg in spec.args.args]
        self.functionKwonlyParams[spec_name] = []
        ast.fix_missing_locations(spec)
        return spec

    def _inject_vararg_specializations(self, node, owner_key=None):
        """Put the specializations owned by `node` where their originals sat.

        A nested def is copied into the function that encloses it rather than
        into the module, so the copy keeps the scope it closes over, and it
        takes the original's position so the names it reads are already bound
        there (#6800).
        """
        specs_by_source = self._vararg_owner_specs.pop(owner_key, None)
        if not specs_by_source:
            return
        specializations = [
            self._vararg_specializations[name] for names in specs_by_source.values()
            for name in names
        ]
        for spec in specializations:
            self.ensure_all_locations(spec)
            ast.fix_missing_locations(spec)
        survivors = [stmt for stmt in node.body if id(stmt) not in self._vararg_dropped_defs]
        body = []
        for stmt in node.body:
            body.extend(self._vararg_specializations[name]
                        for name in specs_by_source.get(id(stmt), ()))
            # A variadic def is dead only once every reference to it has been
            # rewritten; a call this pass cannot lower (`*` unpacking, an alias,
            # a caller in another module) must keep the original, so the
            # frontend reports the unsupported construct rather than a call to a
            # function that no longer exists.
            if (id(stmt) not in self._vararg_dropped_defs
                    or self._name_is_used(survivors + specializations, stmt.name)):
                body.append(stmt)
        node.body = body
