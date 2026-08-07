"""Lowering of functions declaring *args into fixed-arity specializations."""

import ast
import copy


class VarargMixin:

    def _scan_module_vararg_defs(self, node):
        """Mark the variadic defs that live directly in the module body.

        Specializations are injected at module level, so only a def at that
        level can be copied: a nested one would lose the scope it closes over.
        """
        self._vararg_module_defs = {
            id(stmt)
            for stmt in node.body if isinstance(stmt, ast.FunctionDef) and stmt.args.vararg
        }

    def _record_vararg_function(self, node, qualified_name):
        if node.args.vararg:
            self.functionVarargs.add(qualified_name)
            # Registered only once visited, since a specialization is a copy of
            # the body as the converter will see it -- an unvisited body still
            # holds constructs (`for`, comprehensions) with no lowering there.
            if id(node) in self._vararg_module_defs:
                self._vararg_func_defs[node.name] = node
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

        extra = len(node.args) - len(func_def.args.args)
        spec_name = f"{func_def.name}__esbmc_va{extra}"
        if spec_name not in self._vararg_specializations:
            self._vararg_specializations[spec_name] = self._build_vararg_specialization(
                func_def, spec_name, extra)
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

    def _inject_vararg_specializations(self, node):
        if not self._vararg_dropped_defs:
            return
        specializations = list(self._vararg_specializations.values())
        survivors = [stmt for stmt in node.body if id(stmt) not in self._vararg_dropped_defs]
        # A variadic def is dead only once every reference to it has been
        # rewritten; a call this pass cannot lower (`*` unpacking, an alias, a
        # caller in another module) must keep the original, so the frontend
        # reports the unsupported construct rather than a call to a function
        # that no longer exists.
        node.body = [
            stmt for stmt in node.body if id(stmt) not in self._vararg_dropped_defs
            or self._name_is_used(survivors + specializations, stmt.name)
        ]
        for spec in specializations:
            self.ensure_all_locations(spec)
            ast.fix_missing_locations(spec)
        node.body = specializations + node.body
